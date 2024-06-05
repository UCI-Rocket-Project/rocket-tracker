from .component_algos.pid_controller import PIDController
from .utils import TelemetryData
from .environment import Environment
from .component_algos.image_tracker import ImageTracker, NoDetectionError
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Tracker:
    def __init__(self, 
                environment: Environment,
                logger: SummaryWriter
                ):
        k = 1e-3
        self.x_controller = PIDController(k*5,k*1,k*1)
        self.y_controller = PIDController(k*5,k*1,k*1)
        self.environment = environment
        self.logger = logger

        self.img_tracker = ImageTracker()

    
    def start_tracking(self, initial_cam_orientation: tuple[float,float]):
        self.img_tracker.start_new_tracking()

    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData, time: float, control_scope: bool):
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        try:
            pixel_pos = self.img_tracker.estimate_pos(img)[:2]
        except NoDetectionError:
            pixel_pos = None

        x_err = pixel_pos[0] - img.shape[1]//2
        y_err = pixel_pos[1] - img.shape[0]//2
        if not control_scope:
            return 

        input_x = self.x_controller.step(-x_err)
        input_y = self.y_controller.step(-y_err)
        MAX_SLEW_RATE_AZI = 8 
        MAX_SLEW_RATE_ALT = 6
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        y_clipped = np.clip(input_y,-MAX_SLEW_RATE_ALT,MAX_SLEW_RATE_ALT)
        self.environment.move_telescope(x_clipped, y_clipped)
        self.logger.add_scalar("mount/x_input", x_clipped, time*100)
        self.logger.add_scalar("mount/y_input", y_clipped, time*100)
