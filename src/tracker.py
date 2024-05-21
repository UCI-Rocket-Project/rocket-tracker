from .component_algos.pid_controller import PIDController
from .component_algos.launch_detector import LaunchDetector
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from .utils import GroundTruthTrackingData, TelemetryData
from pymap3d import enu2geodetic
from dataclasses import dataclass
from .environment import Environment
from .component_algos.rocket_filter import RocketFilter
from .component_algos.image_tracker import ImageTracker


class Tracker:
    def __init__(self, 
                environment: Environment,
                initial_cam_orientation: tuple[float,float],
                logger: SummaryWriter, 
                ):

        self.camera_res = environment.get_camera_resolution()
        self.focal_len = environment.get_focal_length()
        self.logger = logger
        self.x_controller = PIDController(10,0,1)
        self.y_controller = PIDController(10,0,1)
        self.gps_pos = environment.get_cam_pos_gps() # initial position of mount in GPS coordinates (lat,lng,alt)
        self.environment = environment

        self.initial_cam_orientation = initial_cam_orientation

        self.filter = RocketFilter(environment.get_pad_pos_gps(), environment.get_cam_pos_gps(), initial_cam_orientation)
        self.img_tracker = ImageTracker()
        self.launch_detector: LaunchDetector = None # set in update_tracking on first frame

    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData):
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        pixel_pos = self.img_tracker.estimate_pos(img)
        if pixel_pos is not None:
            if self.launch_detector is None:
                self.launch_detector = LaunchDetector(pixel_pos)

            self.launch_detector.update(pixel_pos)

        if not self.launch_detector.has_detected_launch():
            return None
        


        input_x = self.x_controller.step(az_err)
        input_y = self.y_controller.step(alt_err)
        MAX_SLEW_RATE_AZI = 30 
        MAX_SLEW_RATE_ALT = 30 
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        y_clipped = np.clip(input_y,-MAX_SLEW_RATE_ALT,MAX_SLEW_RATE_ALT)
        self.logger.add_scalar("X Input", x_clipped, global_step)
        self.logger.add_scalar("Y Input", y_clipped, global_step)