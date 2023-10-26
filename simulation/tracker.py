from pid_controller import PIDController

from alpaca.telescope import Telescope, TelescopeAxes
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Tracker:
    def __init__(self, camera_res: tuple[int], logger: SummaryWriter, telescope: Telescope):
        self.camera_res = camera_res
        self.logger = logger
        self.x_controller = PIDController(0.015,0,0.01)
        self.y_controller = PIDController(0.015,0,0.01)
        self.telescope = telescope

    def update_tracking(self, pixel_x: int, pixel_y: int, global_step: int) -> None:
        setpoint_x = self.camera_res[0]//2 
        err_x = setpoint_x  - pixel_x
        setpoint_y = self.camera_res[1]//2
        err_y = setpoint_y - pixel_y

        self.logger.add_scalar("Pixel Tracking Error (X)",err_x,global_step)
        self.logger.add_scalar("Pixel Tracking Error (Y)",err_y,global_step)

        input_x = self.x_controller.step(err_x)
        input_y = self.y_controller.step(err_y)
        x_clipped = np.clip(input_x,-6,6)
        y_clipped = np.clip(input_y,-6,6)
        self.logger.add_scalar("X Input", x_clipped, global_step)
        self.logger.add_scalar("Y Input", y_clipped, global_step)
        self.telescope.MoveAxis(TelescopeAxes.axisSecondary, -y_clipped)
        # self.telescope.MoveAxis(TelescopeAxes.axisPrimary, x_clipped)