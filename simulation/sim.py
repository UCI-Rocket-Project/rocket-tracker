from math import pi, sin, cos
import numpy as np
from time import sleep


from direct.showbase.ShowBase import ShowBase

from direct.task import Task

from direct.actor.Actor import Actor

from direct.interval.IntervalGlobal import Sequence

from panda3d.core import Point3

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

        # Load and transform the panda actor.

        self.rocket_model = self.loader.loadModel("models/panda-model")
        self.rocket_model.setScale(0.005, 0.005, 0.005) # the panda is about 5 meters long
        self.rocket_model.setHpr(0,-90,90)
        self.rocket_model.reparentTo(self.render)

        self.rocket = Rocket()
        self.camera_dist = 1000
        self.camera.setPos(0,-self.camera_dist,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera
        # self.camLens.setFov(0.9)
        self.camLens.setFov(0.9)
        T.SlewToAltAzAsync(0,0)
        while T.Slewing:
            sleep(0.1)
        # 10k feet = 3 km

    def rocketPhysicsTask(self, task):
        self.rocket.step(task.time)
        self.rocket_model.setPos(0,0,self.rocket.height)
        return Task.cont

    # Define a procedure to move the camera.

    def spinCameraTask(self, task):

        # angleDegrees = task.time * 6.0
        angleDegrees = np.rad2deg(np.arctan(self.rocket.height/self.camera_dist))
        print(f"Setpoint: {angleDegrees}, Actual: {T.Altitude}")
        T.SlewToAltAzAsync(0,angleDegrees)
        self.camera.setHpr(T.Azimuth,T.Altitude,0)

        # self.screenshot()

        return Task.cont



app = Sim()

app.run()