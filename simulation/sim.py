from math import pi, sin, cos
import numpy as np


from direct.showbase.ShowBase import ShowBase

from direct.task import Task

from direct.actor.Actor import Actor

from direct.interval.IntervalGlobal import Sequence

from panda3d.core import Point3

from rocket import Rocket



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
        self.rocket.pos = np.array([-1,0,0], dtype=float)
        self.rocket.acc = np.array([0,0,1], dtype=float)

        self.camera.setPos(0,-100,0) # https://docs.panda3d.org/1.10/python/reference/panda3d.core.Camera#panda3d.core.Camera
        self.camLens.setFov(0.9)
        # 10k feet = 3 km

    def rocketPhysicsTask(self, task):
        self.rocket.step(0.1)
        self.rocket_model.setPos(*self.rocket.pos.tolist())
        print(self.rocket.pos[2])
        return Task.cont

    # Define a procedure to move the camera.

    def spinCameraTask(self, task):

        # angleDegrees = task.time * 6.0
        angleDegrees = np.rad2deg(np.arctan(self.rocket.pos[2]/100))


        self.camera.setHpr(0, angleDegrees, 0)

        # self.screenshot()

        return Task.cont



app = Sim()

app.run()