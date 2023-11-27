import numpy as np
import os
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from telescope import Telescope

from direct.showbase.ShowBase import ShowBase

from direct.task import Task
from panda3d.core import SamplerState
from panda3d.core import Shader

from rocket import Rocket
from tracker import Tracker
from utils import GroundTruthTrackingData, TelemetryData, gps_to_enu, enu_to_gps


os.makedirs('runs', exist_ok=True)
num_prev_runs = len(os.listdir('runs')) 
tb_writer = SummaryWriter(f'runs/{num_prev_runs}')
T = Telescope(azimuth=0, altitude=0)

class Sim(ShowBase):

    def __init__(self):

        ShowBase.__init__(self)
        self.setFrameRateMeter(True)

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

        self.camera_dist = 600

        self.rocket_model = self.loader.loadModel("models/rocket.bam")
        self.rocket_model.setScale(0.5, 0.5, 0.5)
        self.rocket_model.setHpr(0,0,0)
        self.rocket_model.setPos(0,self.camera_dist,0)
        self.rocket_model.reparentTo(self.render)

        self.camera.setPos(0,0,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera
        # self.camLens.setFov(0.9)
        self.camera_fov = 9
        self.camera_res = (958, 1078)
        self.cam_focal_len_pixels = self.camera_res[0]/(2*np.tan(np.deg2rad(self.camera_fov/2)))
        self.camLens.setFov(self.camera_fov)

        self.rocket = Rocket(np.array([0,self.camera_dist,0]))
        # get tracker position in gps coordinates based on rocket telemetry
        telem: TelemetryData = self.rocket.get_telemetry(0)
        tracker_pos_gps = enu_to_gps(np.array([0,-self.camera_dist,0]), np.array([telem.gps_lat, telem.gps_lng, telem.altimeter_reading]))
        self.tracker = Tracker(self.camera_res, self.cam_focal_len_pixels, tb_writer, T, self.rocket.position, tracker_pos_gps)

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.rocketPhysicsTask, "Physics")
        self.prev_rocket_position = None
        self.prev_rocket_observation_time = None 

    def rocketPhysicsTask(self, task):
        rocket_pos = self.rocket.get_position(task.time)
        x,y,z = rocket_pos
        self.rocket_model.setPos(x,y,z)
        tb_writer.add_scalar("Rocket X Position", x, task.time*100)
        tb_writer.add_scalar("Rocket Y Position", y, task.time*100)
        tb_writer.add_scalar("Rocket Z Position", z, task.time*100)
        if self.prev_rocket_observation_time is not None:
            az_old, alt_old = self.getGroundTruthAzAlt(self.prev_rocket_position)
            az_new, alt_new = self.getGroundTruthAzAlt(rocket_pos)
            tb_writer.add_scalar("Rocket Azimuth", az_new, task.time*100)
            tb_writer.add_scalar("Rocket Altitude", alt_new, task.time*100)
            az_derivative = (az_new-az_old)/(task.time-self.prev_rocket_observation_time)
            alt_derivative = (alt_new-alt_old)/(task.time-self.prev_rocket_observation_time)
            tb_writer.add_scalar("Rocket Azimuth Derivative", az_derivative, task.time*100)
            tb_writer.add_scalar("Rocket Altitude Derivative", alt_derivative, task.time*100)
        self.prev_rocket_position = rocket_pos
        self.prev_rocket_observation_time = task.time
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
        # blocks out the FPS counter by copying the part of the background in the bottom right to the top right.
        img[:50, -150:, :] = img[-50:,-150:,:]
        return img
    
    def getGroundTruthRocketPixelCoordinates(self, time):
        alt, az = np.deg2rad(T.Altitude), np.deg2rad(T.Azimuth)
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

        rocket_pos = self.rocket.get_position(time)

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

    def spinCameraTask(self, task):
        # angleDegrees = task.time * 6.0
        img = self.getImage()
        if img is None:
            return Task.cont

        x,y = self.getGroundTruthRocketPixelCoordinates(task.time)
        
        ground_truth_tracking_data = GroundTruthTrackingData(
            pixel_coordinates = (x,y),
            enu_coordinates = self.rocket.position,
            az_alt = self.getGroundTruthAzAlt(self.rocket.position)
        )

        self.tracker.update_tracking(
            img,
            self.rocket.get_telemetry(task.time),
            task.time*100, 
            ground_truth_tracking_data
        )

        self.camera.setHpr(T.Azimuth,T.Altitude,0)
        tb_writer.add_scalar("Telescope Azimuth", T.Azimuth, task.time*100)
        tb_writer.add_scalar("Telescope Altitude", T.Altitude, task.time*100)

        # cv.circle(img, [x,y], 10, (255,0,0), -1)
        # cv.imwrite("latest.png", img) 
        return Task.cont



app = Sim()

app.run()