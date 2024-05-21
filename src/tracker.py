from .component_algos.pid_controller import PIDController
from .component_algos.launch_detector import LaunchDetector
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .utils import TelemetryData
from .environment import Environment
from .component_algos.rocket_filter import RocketFilter
from .component_algos.image_tracker import ImageTracker, NoKeypointsFoundError
import pymap3d as pm


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
        self.step_value = 0

    def _pixel_pos_to_az_alt(self, pixel_pos: np.ndarray) -> tuple[float,float]:
        raise NotImplementedError()

    def _ecef_to_az_alt(self, ecef_pos: np.ndarray) -> tuple[float,float]:
        raise NotImplementedError()

    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData, time: float):
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        try:
            pixel_pos = self.img_tracker.estimate_pos(img)
        except NoKeypointsFoundError:
            pixel_pos = None

        if pixel_pos is not None:
            if self.launch_detector is None:
                self.launch_detector = LaunchDetector(pixel_pos)

            if not self.launch_detector.has_detected_launch():
                self.launch_detector.update(pixel_pos)

        if not self.launch_detector.has_detected_launch():
            return
        
        # calculate azimuth and altitude based on pixel position

        if pixel_pos is not None:
            az, alt = self._pixel_pos_to_az_alt(pixel_pos)
            self.filter.predict_update_bearing(time - self.launch_detector.get_launch_time(), np.array([az, alt]))

        if telem_measurements is not None:
            ecef_pos = pm.geodetic2ecef(telem_measurements.gps_lat, telem_measurements.gps_lng, telem_measurements.altimeter_reading)
            alt = telem_measurements.altimeter_reading
            z = np.array([*ecef_pos, alt])
            self.filter.predict_update_telem(time - self.launch_detector.get_launch_time(), z)

        ecef_pos = self.filter.x
        az, alt = self._ecef_to_az_alt(ecef_pos)
        current_az, current_alt = self.environment.get_telescope_orientation()

        az_err = az - current_az
        alt_err = alt - current_alt

        input_x = self.x_controller.step(az_err)
        input_y = self.y_controller.step(alt_err)
        MAX_SLEW_RATE_AZI = 30 
        MAX_SLEW_RATE_ALT = 30 
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        y_clipped = np.clip(input_y,-MAX_SLEW_RATE_ALT,MAX_SLEW_RATE_ALT)
        self.logger.add_scalar("X Input", x_clipped, self.step_value, walltime=time)
        self.logger.add_scalar("Y Input", y_clipped, self.step_value, walltime=time)
        self.step_value += 1