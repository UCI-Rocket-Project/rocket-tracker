import numpy as np
import os
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from .sim_telescope import SimTelescope

from direct.showbase.ShowBase import ShowBase

from direct.task import Task
from panda3d.core import lookAt, Quat, Shader, SamplerState, Vec3, NodePath

from .rocket import Rocket
from src.utils import TelemetryData
from src.environment import Environment
from pymap3d import enu2geodetic, ecef2enu, enu2ecef
from src.joystick_commander import JoystickCommander
from src.component_algos.depth_of_field import DOFCalculator, MM_PER_PIXEL
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
        self.launch_time = 5
        self.rocket = Rocket(self.pad_geodetic_pos, self.launch_time)
        rocket_pos_enu = np.array(ecef2enu(*self.rocket.get_position_ecef(0), *self.cam_geodetic_location))
        self.rocket_model.setPos(rocket_pos_enu[0] - 2, *rocket_pos_enu[1:])
        # get tracker position in gps coordinates based on rocket telemetry
        telem: TelemetryData = self.rocket.get_telemetry(0)
        self.telem = telem

        self.focus_plane_position = 0
        self.focus_offset = 0 # distance from lens to focus plane is this number plus the focal length
        cam_fstop = 7

        class SimulationEnvironment(Environment):
            def __init__(env_self):
                super().__init__(self.pad_geodetic_pos, self.cam_geodetic_location, camera_res, focal_len_pixels, cam_fstop)

            def get_telescope_orientation(env_self) -> tuple[float, float]:
                return self.telescope.read_position()

            def get_telescope_speed(env_self) -> tuple[float,float]:
                return self.telescope.read_speed()

            def move_telescope(env_self, v_azimuth: float, v_altitude: float):
                self.telescope.slew_rate_azi_alt(v_azimuth, v_altitude)

            def get_camera_image(env_self) -> np.ndarray:
                img = self.getImage()
                if img is None:
                    return None
                rocket_bbox = self.getRocketBoundingBox()
                min_x, min_y, max_x, max_y = rocket_bbox
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(img.shape[1], max_x)
                max_y = min(img.shape[0], max_y)


                rocket_to_cam_distance = np.linalg.norm(ecef2enu(*self.rocket.get_position_ecef(self.telem.time), *self.cam_geodetic_location))

                focal_len_mm = focal_len_pixels * MM_PER_PIXEL
                dof_calculator = DOFCalculator.from_fstop(focal_len_mm, cam_fstop)
                circle_of_confusion_mm = dof_calculator.circle_of_confusion(rocket_to_cam_distance, self.focus_offset + focal_len_mm)
                circle_of_confusion_pixels = circle_of_confusion_mm/MM_PER_PIXEL
                self.logger.add_scalar("rendering/circle_of_confusion", circle_of_confusion_pixels, self.telem.time*100)
                self.logger.add_scalar("rendering/distance_to_cam", rocket_to_cam_distance, self.telem.time*100)

                # apply blur but only to the rocket
                if circle_of_confusion_pixels >= 1 and max_x-min_x > 0 and max_y-min_y > 0:
                    kernel_size = int(np.ceil(circle_of_confusion_pixels))//2*2+1 # make sure it's odd
                    img_slice = img[min_y:max_y, min_x:max_x]
                    cv.blur(src=img_slice,dst=img_slice, ksize=(kernel_size, kernel_size))
                return img

            # def get_ground_truth_pixel_loc(env_self, time: float) -> tuple[int,int]:
            #     return self.getGroundTruthRocketPixelCoordinates(time)

            def move_focuser(env_self, position: int):
                lower_bound, upper_bound = env_self.get_focuser_bounds()
                assert lower_bound <= position <= upper_bound, f'Focuser position {position} is out of bounds [{lower_bound}, {upper_bound}]'
                self.focus_offset = position 

            def get_focuser_bounds(env_self) -> tuple[int,int]:
                return 0, 60

            def get_focuser_position(env_self) -> int:
                return self.focus_offset

            def get_telemetry(env_self) -> TelemetryData:
                return self.telem # updated in rocketPhysicsTask
            
        estimate_logger = SummaryWriter("runs/estimate")
        
        self.controller = JoystickCommander(SimulationEnvironment(), estimate_logger)

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

            pixel_x, pixel_y = self.getGroundTruthRocketPixelCoordinates()
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
        return img

    def _get_coord_pixel_loc(self, coord: np.ndarray) -> tuple[int,int]:
        '''
        Coord is in 3d ENU frame relative to self.cam_geodetic_location
        '''
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
    
        rocket_cam_pos = alt_rotation.T @ az_rotation.T @ coord

        w,h = self.camera_res

        focal_len_pixels = self.cam_focal_len_pixels

        pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
        pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

        return int(pixel_x), int(pixel_y)

    def getGroundTruthRocketPixelCoordinates(self):
        pt = self.rocket_model.getPos()
        rocket_pos = np.array([pt.x, pt.y, pt.z])
        return self._get_coord_pixel_loc(rocket_pos)

    def getRocketBoundingBox(self):
        '''
        This method is approximate and only guarantees the bounding box is on the rocket, not that it completely covers the rocket.
        Returns bounding box in pixels (min_x, min_y, max_x, max_y). Doesn't guarantee that the bounding box is entirely in the image.
        '''
        pt = self.rocket_model.getPos()
        rocket_center_pos = np.array([pt.x, pt.y, pt.z])
        rocket_vel_enu = self.rocket.get_velocity(self.telem.time) # this is slightly off but close enough for veloctiy

        # reference points are the top and bottom of the rocket plus or minus the rocket radius
        # this will assume that the rocket is pointed in the direction of its velocity vector
        rocket_radius = 0.8
        rocket_height = 7
        vel_direction = rocket_vel_enu/np.linalg.norm(rocket_vel_enu) if np.linalg.norm(rocket_vel_enu) > 1e-1 else np.array([0,0,1])
        top_point = rocket_center_pos + rocket_height/2 * vel_direction
        bottom_point = rocket_center_pos - rocket_height/2 * vel_direction

        orthogonal_direction = np.cross(vel_direction, np.array([1,0,0])) # arbitrary vector that is orthogonal to the velocity vector
        orthogonal_direction_2 = np.cross(vel_direction, orthogonal_direction) # second orthogonal vector, also orthogonal to the first

        top_4_points = [
            top_point + rocket_radius * orthogonal_direction,
            top_point - rocket_radius * orthogonal_direction,
            top_point + rocket_radius * orthogonal_direction_2,
            top_point - rocket_radius * orthogonal_direction_2
        ]

        bottom_4_points = [
            bottom_point + rocket_radius * orthogonal_direction,
            bottom_point - rocket_radius * orthogonal_direction,
            bottom_point + rocket_radius * orthogonal_direction_2,
            bottom_point - rocket_radius * orthogonal_direction_2
        ]

        all_points = np.array(list.__add__(top_4_points,bottom_4_points)) # 8 points, using __add__ to make it clear it's not element-wise

        pixel_coords = np.array([self._get_coord_pixel_loc(point) for point in all_points])

        min_x, min_y = np.min(pixel_coords, axis=0)
        max_x, max_y = np.max(pixel_coords, axis=0)
        return min_x, min_y, max_x, max_y

    def getGroundTruthAzAlt(self, rocket_pos: np.ndarray) -> tuple[float,float]:
        az = -np.arctan2(rocket_pos[0], rocket_pos[1])
        alt = np.arctan2(rocket_pos[2], np.sqrt(rocket_pos[0]**2 + rocket_pos[1]**2))
        return np.rad2deg(az), np.rad2deg(alt)
    
    def _get_img_debug_callback(self, time):
        pixel_x, pixel_y = self.getGroundTruthRocketPixelCoordinates()
        def callback(img: np.ndarray):
            cv.circle(img, (pixel_x, pixel_y), 10, (255,255,255), -1)
            cv.putText(img, 'Rocket', (pixel_x, pixel_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            rocket_bbox = self.getRocketBoundingBox()
            cv.rectangle(img, (rocket_bbox[0], rocket_bbox[1]), (rocket_bbox[2], rocket_bbox[3]), (255,255,255), 2)
        return callback

    def spinCameraTask(self, task):
        t_azi, t_alt  = self.telescope.read_position()
        self.camera.setHpr(t_azi, t_alt,0)
        self.controller.loop_callback(task.time, self._get_img_debug_callback(task.time))
        return Task.cont



app = Sim()

app.run()