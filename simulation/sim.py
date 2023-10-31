import numpy as np
import os
from time import sleep
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from alpaca.telescope import Telescope

from direct.showbase.ShowBase import ShowBase

from direct.task import Task

from rocket import Rocket
from tracker import Tracker


os.makedirs('runs', exist_ok=True)
num_prev_runs = len(os.listdir('runs')) 
tb_writer = SummaryWriter(f'runs/{num_prev_runs}')
T = Telescope('localhost:32323', 0) # Local Omni Simulator

class Sim(ShowBase):

    def __init__(self):

        ShowBase.__init__(self)


        # Disable the camera trackball controls.

        self.disableMouse()

        self.camera_dist = 600

        self.rocket_model = self.loader.loadModel("models/panda-model")
        self.rocket_model.setScale(0.005, 0.005, 0.005) # the panda is about 5 meters long after this scaling
        self.rocket_model.setHpr(0,-90,90)
        self.rocket_model.setPos(0,self.camera_dist,0)
        self.rocket_model.reparentTo(self.render)

        self.rocket = Rocket(np.array([0,self.camera_dist,0]))
        self.camera.setPos(0,0,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera
        # self.camLens.setFov(0.9)
        self.camera_fov = 0.9
        self.camLens.setFov(self.camera_fov)
        self.camera_res = (958, 1078)
        T.AbortSlew()
        T.SlewToAltAzAsync(0,0)
        while T.Slewing:
            sleep(0.1)
        self.tracker = Tracker(self.camera_res, tb_writer, T)

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.rocketPhysicsTask, "Physics")

    def rocketPhysicsTask(self, task):
        self.rocket.step(task.time)
        x,y,z = self.rocket.position
        self.rocket_model.setPos(x,y,z)
        tb_writer.add_scalar("Rocket X Position", x, task.time)
        tb_writer.add_scalar("Rocket Y Position", y+self.camera_dist, task.time)
        tb_writer.add_scalar("Rocket Z Position", z, task.time)
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
    
    def getGroundTruthRocketPixelCoordinates(self):
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

        rocket_pos = self.rocket.position

        rocket_cam_pos = alt_rotation.T @ az_rotation.T @ rocket_pos

        w,h = self.camera_res

        focal_len_pixels = w/(2*np.tan(np.deg2rad(self.camera_fov/2)))

        pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
        pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

        return int(pixel_x), int(pixel_y)

    def spinCameraTask(self, task):

        # angleDegrees = task.time * 6.0
        img = self.getImage()
        if img is None:
            return Task.cont
        x,y = self.getGroundTruthRocketPixelCoordinates()
        
        pos = self.tracker.update_image_tracking(img)
        if pos is not None:
            tb_writer.add_scalar("X Estimation Error", pos[0]-x, task.time)
            tb_writer.add_scalar("Y Estimation Error", pos[1]-y, task.time)
        if pos is None:
            return Task.cont
        self.tracker.update_tracking(pos[0],pos[1],task.time)

        self.camera.setHpr(T.Azimuth,T.Altitude,0)
        tb_writer.add_scalar("Azimuth", T.Azimuth, task.time)
        tb_writer.add_scalar("Altitude", T.Altitude, task.time)

        cv.circle(img, [x,y], 10, (255,0,0), -1)
        cv.imwrite("latest.png", img) 
        return Task.cont



app = Sim()

app.run()