import numpy as np
import os
from time import sleep
import cv2 as cv
import matplotlib.pyplot as plt
from pid_controller import PIDController
from torch.utils.tensorboard import SummaryWriter

from direct.showbase.ShowBase import ShowBase

from direct.task import Task

from direct.actor.Actor import Actor

from direct.interval.IntervalGlobal import Sequence

from panda3d.core import Point3 
from panda3d.physics import ActorNode

from rocket import Rocket

from alpaca.telescope import Telescope, TelescopeAxes

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

        self.rocket = Rocket()
        self.camera.setPos(0,0,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera
        # self.camLens.setFov(0.9)
        self.camera_fov = 0.9
        self.camLens.setFov(self.camera_fov)
        self.camera_res = (958, 1078)
        T.AbortSlew()
        T.SlewToAltAzAsync(0,0.5)
        while T.Slewing:
            sleep(0.1)
        # 10k feet = 3 km

        self.x_controller = PIDController(0.005,0.001,0.01)
        self.y_controller = PIDController(0.015,0,0.01)

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        self.taskMgr.add(self.rocketPhysicsTask, "Physics")

    def rocketPhysicsTask(self, task):
        self.rocket.step(task.time)
        x,y,z = self.rocket.position
        self.rocket_model.setPos(x,y+self.camera_dist,z)
        tb_writer.add_scalar("Rocket X Position", x, task.frame)
        tb_writer.add_scalar("Rocket Y Position", y+self.camera_dist, task.frame)
        tb_writer.add_scalar("Rocket Z Position", z, task.frame)
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

        rocket_pos = np.array(self.rocket.position)+np.array([0,self.camera_dist,0])

        rocket_cam_pos = alt_rotation.T @ az_rotation.T @ rocket_pos

        w,h = self.camera_res

        focal_len_pixels = w/(2*np.tan(np.deg2rad(self.camera_fov/2)))

        pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
        pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

        return int(pixel_x), int(pixel_y)

    def spinCameraTask(self, task):

        # angleDegrees = task.time * 6.0
        x,y = self.getGroundTruthRocketPixelCoordinates()
        setpoint_x = self.camera_res[0]//2 
        err_x = setpoint_x  - x
        setpoint_y = self.camera_res[1]//2
        err_y = setpoint_y - y

        tb_writer.add_scalar("Pixel Tracking Error (X)",err_x,task.frame)
        tb_writer.add_scalar("Pixel Tracking Error (Y)",err_y,task.frame)

        # T.SlewToAltAzAsync(0,angleDegrees)
        input_x = self.x_controller.step(err_x)
        input_y = self.y_controller.step(err_y)
        x_clipped = np.clip(input_x,-6,6)
        y_clipped = np.clip(input_y,-6,6)
        tb_writer.add_scalar("X Input", x_clipped, task.frame)
        tb_writer.add_scalar("Y Input", y_clipped, task.frame)
        # print(control_input, setpoint, curr, clipped_input)
        T.MoveAxis(TelescopeAxes.axisSecondary, -y_clipped)
        # T.MoveAxis(TelescopeAxes.axisPrimary, -x_clipped)

        self.camera.setHpr(T.Azimuth,T.Altitude,0)
        img = self.getImage()
        if img is None:
            return Task.cont
        cv.circle(img, [x,y], 10, (255,0,0), -1)
        cv.imwrite("latest.png", img) 
        return Task.cont



app = Sim()

app.run()