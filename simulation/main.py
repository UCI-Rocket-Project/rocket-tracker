import numpy as np
import os
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from .sim_telescope import SimTelescope

from direct.showbase.ShowBase import ShowBase

from direct.task import Task
from panda3d.core import lookAt, Quat, Shader, SamplerState, Vec3

from .rocket import Rocket
from src.tracker import Tracker
from src.utils import GroundTruthTrackingData, TelemetryData
from src.environment import Environment
from pymap3d import geodetic2enu, enu2geodetic, ecef2enu
from src.joystick_controller import JoystickController
import shutil
print("Removing old runs directory")
if os.path.exists("runs"):
    shutil.rmtree("runs")
os.makedirs("runs")

class Sim(ShowBase):

    def __init__(self):

        ShowBase.__init__(self)

        self.skybox = self.loader.loadModel("models/skybox.bam")
        self.skybox.reparentTo(self.render)
        self.skybox.set_scale(20000)

        skybox_texture = self.loader.loadTexture("models/desert_sky_2k.jpg")
        skybox_texture.set_minfilter(SamplerState.FT_linear)
        skybox_texture.set_magfilter(SamplerState.FT_linear)
        skybox_texture.set_wrap_u(SamplerState.WM_repeat)
        skybox_texture.set_wrap_v(SamplerState.WM_mirror)
        skybox_texture.set_anisotropic_degree(16)
        self.skybox.set_texture(skybox_texture)
        # Disable the camera trackball controls.

        skybox_shader = Shader.load(Shader.SL_GLSL, "skybox.vert.glsl", "skybox.frag.glsl")
        self.skybox.set_shader(skybox_shader)

        self.disableMouse()

        self.rocket_model = self.loader.loadModel("models/rocket.bam")
        self.rocket_model.setScale(0.5, 0.5, 0.5)
        self.rocket_model.setHpr(0,0,0)
        self.rocket_model.reparentTo(self.render)

        self.camera.setPos(0,0,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera

        self.camera_fov = 0.9 # degrees
        camera_res = (958, 1078)
        self.camera_res = camera_res
        focal_len_pixels = self.camera_res[0]/(2*np.tan(np.deg2rad(self.camera_fov/2)))
        self.cam_focal_len_pixels = focal_len_pixels
        self.camLens.setFov(self.camera_fov)
        self.telescope = SimTelescope(-69.62, 0)
        # camera is at (0,0,0) but that's implicit

        self.pad_geodetic_pos = np.array([35.347104, -117.808953, 620])
        self.cam_geodetic_location = np.array([35.34222222, -117.82500000, 620])

        self.logger = SummaryWriter("runs/ground_truth")
        self.launch_time = 10
        self.rocket = Rocket(self.pad_geodetic_pos, self.launch_time)
        # get tracker position in gps coordinates based on rocket telemetry
        telem: TelemetryData = self.rocket.get_telemetry(0)
        self.telem = telem

        class SimulationEnvironment(Environment):
            def __init__(env_self):
                super().__init__(self.pad_geodetic_pos, self.cam_geodetic_location, camera_res, focal_len_pixels)

            def get_telescope_orientation(env_self) -> tuple[float, float]:
                return self.telescope.read_position()

            def get_telescope_speed(env_self) -> tuple[float,float]:
                return self.telescope.read_speed()

            def move_telescope(env_self, v_azimuth: float, v_altitude: float):
                self.telescope.slew_rate_azi_alt(v_azimuth, v_altitude)

            def get_camera_image(env_self) -> np.ndarray:
                return self.getImage()

            def get_ground_truth_pixel_loc(env_self, time: float) -> tuple[int,int]:
                return self.getGroundTruthRocketPixelCoordinates(time)

            def get_telemetry(env_self) -> TelemetryData:
                return self.telem # updated in rocketPhysicsTask
            
        estimate_logger = SummaryWriter("runs/estimate")
        
        self.controller = JoystickController(SimulationEnvironment(), estimate_logger)

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.rocketPhysicsTask, "Physics")
        self.prev_rocket_position = None
        self.prev_rocket_observation_time = None 


    def rocketPhysicsTask(self, task):
        rocket_pos_ecef = self.rocket.get_position_ecef(task.time)
        rocket_vel = self.rocket.get_velocity(task.time)
        rocket_accel = self.rocket.get_acceleration(task.time)
        
        rocket_pos_enu = np.array(ecef2enu(*rocket_pos_ecef, *self.cam_geodetic_location))
        self.rocket_model.setPos(*rocket_pos_enu)
        if self.prev_rocket_position is not None:
            quat = Quat()
            # if difference is low enough, just look straight up. Without this, it flips around at the start of the flight
            if np.linalg.norm(rocket_pos_enu-self.prev_rocket_position) < 1e-1:
                lookAt(quat, Vec3(0,1,0),Vec3(0,0,0))
            else:
                lookAt(quat, Vec3(*self.prev_rocket_position),Vec3(*rocket_pos_enu))
            self.rocket_model.setQuat(quat)
        x,y,z = rocket_pos_enu
        self.logger.add_scalar("enu position/x", x, task.time*100)
        self.logger.add_scalar("enu position/y", y, task.time*100)
        self.logger.add_scalar("enu position/z", z, task.time*100)

        self.logger.add_scalar("enu velocity/x", rocket_vel[0], task.time*100)
        self.logger.add_scalar("enu velocity/y", rocket_vel[1], task.time*100)
        self.logger.add_scalar("enu velocity/z", rocket_vel[2], task.time*100)

        self.logger.add_scalar("enu accel/x", rocket_accel[0], task.time*100)
        self.logger.add_scalar("enu accel/y", rocket_accel[1], task.time*100)
        self.logger.add_scalar("enu accel/z", rocket_accel[2], task.time*100)
        if self.prev_rocket_observation_time is not None:
            az_old, alt_old = self.getGroundTruthAzAlt(self.prev_rocket_position)
            az_new, alt_new = self.getGroundTruthAzAlt(rocket_pos_enu)
            self.logger.add_scalar("bearing/azimuth", az_new, task.time*100)
            self.logger.add_scalar("bearing/altitude", alt_new, task.time*100)

            pixel_x, pixel_y = self.getGroundTruthRocketPixelCoordinates(task.time)
            self.logger.add_scalar("pixel position/x", pixel_x, task.time*100)
            self.logger.add_scalar("pixel position/y", pixel_y, task.time*100)
            # az_derivative = (az_new-az_old)/(task.time-self.prev_rocket_observation_time)
            # alt_derivative = (alt_new-alt_old)/(task.time-self.prev_rocket_observation_time)
        launched = int(task.time>=self.launch_time)
        self.telescope.step(task.time)

        geodetic_pos = enu2geodetic(*rocket_pos_enu, *self.cam_geodetic_location)
        self.logger.add_scalar("telemetry/lat", geodetic_pos[0], task.time*100)
        self.logger.add_scalar("telemetry/lng", geodetic_pos[1], task.time*100)
        self.logger.add_scalar("telemetry/alt", geodetic_pos[2], task.time*100)

        self.logger.add_scalar("launched", launched, task.time*100)
        self.prev_rocket_position = rocket_pos_enu
        self.prev_rocket_observation_time = task.time
        self.telem = self.rocket.get_telemetry(task.time)
        return Task.cont

    def getImage(self):
        self.screenshot()
        files_list = os.listdir(".")
        img = None
        for f in files_list:
            if f.endswith(".jpg"):
                img = cv.imread(f)
                os.remove(f)
                break
        if img is None:
            return
        return img
    
    def getGroundTruthRocketPixelCoordinates(self, time):
        t_azi, t_alt  = self.telescope.read_position()
        az, alt = np.deg2rad(t_azi), np.deg2rad(t_alt)
        alt_rotation = np.array([
            [1,  0,  0],
            [0,  np.cos(alt),  -np.sin(alt)],
            [0,  np.sin(alt), np.cos(alt)] 
        ])

        az_rotation = np.array([
            [np.cos(az),  -np.sin(az),  0],
            [np.sin(az),  np.cos(az),  0],
            [0,  0,  1]
        ])

        rocket_pos_ecef = self.rocket.get_position_ecef(time)
        rocket_pos = np.array(ecef2enu(*rocket_pos_ecef, *self.cam_geodetic_location))

        rocket_cam_pos = alt_rotation.T @ az_rotation.T @ rocket_pos

        w,h = self.camera_res

        focal_len_pixels = w/(2*np.tan(np.deg2rad(self.camera_fov/2)))

        pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
        pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

        return int(pixel_x), int(pixel_y)

    def getGroundTruthAzAlt(self, rocket_pos: np.ndarray) -> tuple[float,float]:
        az = -np.arctan2(rocket_pos[0], rocket_pos[1])
        alt = np.arctan2(rocket_pos[2], np.sqrt(rocket_pos[0]**2 + rocket_pos[1]**2))
        return np.rad2deg(az), np.rad2deg(alt)
    
    def _get_img_debug_callback(self, time):
        pixel_x, pixel_y = self.getGroundTruthRocketPixelCoordinates(time)
        def callback(img: np.ndarray):
            cv.circle(img, (pixel_x, pixel_y), 10, (255,0,0), -1)
        return callback

    def spinCameraTask(self, task):
        t_azi, t_alt  = self.telescope.read_position()
        self.camera.setHpr(t_azi, t_alt,0)
        self.controller.loop_callback(task.time, self._get_img_debug_callback(task.time))
        return Task.cont



app = Sim()

app.run()