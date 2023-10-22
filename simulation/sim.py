import numpy as np
import os
from time import sleep
import cv2 as cv


from direct.showbase.ShowBase import ShowBase

from direct.task import Task

from direct.actor.Actor import Actor

from direct.interval.IntervalGlobal import Sequence

from panda3d.core import Point3 
from panda3d.physics import ActorNode

from rocket import Rocket

from alpaca.telescope import Telescope, TelescopeAxes


T = Telescope('localhost:32323', 0) # Local Omni Simulator

class Sim(ShowBase):

    def __init__(self):

        ShowBase.__init__(self)


        # Disable the camera trackball controls.

        self.disableMouse()


        # Load the environment model.

        # self.scene = self.loader.loadModel("models/environment")

        # # Reparent the model to render.

        # self.scene.reparentTo(self.render)

        # # Apply scale and position transforms on the model.

        # self.scene.setScale(0.25, 0.25, 0.25)

        # self.scene.setPos(-8, 42, 0)


        # Add the spinCameraTask procedure to the task manager.

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        self.taskMgr.add(self.rocketPhysicsTask, "Physics")

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
        T.SlewToAltAzAsync(0,0)
        while T.Slewing:
            sleep(0.1)
        # 10k feet = 3 km
        self.total_err = 0
        self.prev_err = 0
        self.setpoints = []

    def rocketPhysicsTask(self, task):
        self.rocket.step(task.time)
        self.rocket_model.setPos(0,self.camera_dist,self.rocket.height)
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

        rocket_pos = np.array([0,self.camera_dist,self.rocket.height])

        rocket_cam_pos = alt_rotation.T @ az_rotation.T @ rocket_pos

        w,h = self.camera_res

        focal_len_pixels = w/(2*np.tan(np.deg2rad(self.camera_fov/2)))

        pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
        pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

        return int(pixel_x), int(pixel_y)

    def spinCameraTask(self, task):

        # angleDegrees = task.time * 6.0
        setpoint = np.rad2deg(np.arctan(self.rocket.height/self.camera_dist))
        curr = T.Altitude
        err = setpoint - curr
        d = err - self.prev_err 
        self.prev_err = err
        self.total_err+=err
        KP, KI, KD = 10,0,1
        # T.SlewToAltAzAsync(0,angleDegrees)
        control_input = KP*err + KI*self.total_err - KD * d
        clipped_input = np.clip(control_input,-6,6)
        # print(control_input, setpoint, curr, clipped_input)
        T.MoveAxis(TelescopeAxes.axisSecondary, -clipped_input)
        self.camera.setHpr(T.Azimuth,T.Altitude,0)
        img = self.getImage()
        if img is None:
            return Task.cont
        x,y = self.getGroundTruthRocketPixelCoordinates()
        print(x,y)
        cv.circle(img, [x,y], 10, (255,0,0), -1)
        cv.imwrite("latest.png", img) 
        return Task.cont



app = Sim()

app.run()