from src.component_algos.pid_controller import PIDController
from src.component_algos.mpc_controller import MPCController
from src.component_algos.launch_detector import LaunchDetector
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.utils import TelemetryData
from src.environment import Environment
from src.component_algos.rocket_filter import RocketFilter
from src.component_algos.image_tracker import ImageTracker, NoDetectionError
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
import cv2 as cv
from src.component_algos.depth_of_field import DOFCalculator, MM_PER_PIXEL


class Tracker:
    def __init__(self, 
                environment: Environment,
                logger: SummaryWriter, 
                ):

        self.camera_res = environment.get_camera_resolution()
        self.focal_len_pixels = environment.get_focal_length_pixels()
        self.logger = logger
        self.x_controller = PIDController(5,1,1)
        self.y_controller = PIDController(5,1,1)
        self.mpc_controller = MPCController()
        self.gps_pos = environment.get_cam_pos_gps() # initial position of mount in GPS coordinates (lat,lng,alt)
        self.environment = environment
        focal_len_mm = self.focal_len_pixels * MM_PER_PIXEL
        print(f'Focal length: {focal_len_mm}mm')
        self.dof_calc = DOFCalculator.from_fstop(focal_len_mm, environment.cam_fstop)

        self.img_tracker = ImageTracker()
        self.active_tracking = False

    def _pixel_pos_to_az_alt(self, pixel_pos: np.ndarray) -> tuple[float,float]:
        az = -np.arctan2(pixel_pos[0] - self.camera_res[0] / 2, self.focal_len_pixels)
        alt = -np.arctan2(pixel_pos[1] - self.camera_res[1] / 2, self.focal_len_pixels)
        pixel_rot = R.from_euler("ZY", [az, alt], degrees=False)
        initial_rot = R.from_euler("ZY", self.environment.get_telescope_orientation(), degrees=True)

        final_rot = initial_rot * pixel_rot
        az, alt = final_rot.as_euler("ZYX", degrees=True)[:2]
        return az, alt

    def _ecef_to_az_alt(self, ecef_pos: np.ndarray) -> tuple[float,float]:
        return self.filter.hx_bearing(ecef_pos)
    
    def start_tracking(self, initial_cam_orientation: tuple[float,float]):
        self.initial_cam_orientation = initial_cam_orientation
        self.filter = RocketFilter(self.environment.get_pad_pos_gps(), self.gps_pos, self.initial_cam_orientation, writer=SummaryWriter("runs/filter"))
        self.launch_detector = None
        self.active_tracking = True
        self.img_tracker.start_new_tracking()

    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData, time: float, control_scope: bool):
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        if hasattr(self.environment, "get_ground_truth_pixel_loc"):
            pixel_pos = self.environment.get_ground_truth_pixel_loc(time)
        else:
            try:
                pixel_pos = self.img_tracker.estimate_pos(img)[:2]
            except NoDetectionError:
                pixel_pos = None
        cv.circle(img, pixel_pos, 10, (0,255,0), 2)

        if not self.active_tracking:
            return
        
        if pixel_pos is not None:
            self.logger.add_scalar("pixel position/x", pixel_pos[0], time*100)
            self.logger.add_scalar("pixel position/y", pixel_pos[1], time*100)
            if self.launch_detector is None:
                self.launch_detector = LaunchDetector(pixel_pos)

            if not self.launch_detector.has_detected_launch():
                self.launch_detector.update(pixel_pos, time)

        if self.launch_detector is None:
            raise RuntimeError("Need to see rocket in first frame to start tracking, no rocket found.")

        if not self.launch_detector.has_detected_launch():
            self.logger.add_scalar("launched", 0, time*100)
            return
        self.logger.add_scalar("launched", 1, time*100)

        # calculate azimuth and altitude based on pixel position

        if pixel_pos is not None:
            az, alt = self._pixel_pos_to_az_alt(pixel_pos)
            self.filter.predict_update_bearing(time - self.launch_detector.get_launch_time(), np.array([az, alt]))

        if telem_measurements is not None:
            ecef_pos = pm.geodetic2ecef(telem_measurements.gps_lat, telem_measurements.gps_lng, telem_measurements.altimeter_reading)
            alt = telem_measurements.altimeter_reading
            z = np.array([*ecef_pos, alt])
            self.logger.add_scalar("telemetry/lat", telem_measurements.gps_lat, time*100)
            self.logger.add_scalar("telemetry/lng", telem_measurements.gps_lng, time*100)
            self.logger.add_scalar("telemetry/alt", telem_measurements.altimeter_reading, time*100)
            self.filter.predict_update_telem(time - self.launch_detector.get_launch_time(), z)

        ecef_pos = self.filter.x[:3]
        target_az, target_alt = self._ecef_to_az_alt(ecef_pos)
        current_az, current_alt = self.environment.get_telescope_orientation()

        enu_pos = pm.ecef2enu(*ecef_pos, *self.environment.get_cam_pos_gps())
        self.logger.add_scalar("enu position/x", enu_pos[0], time*100)
        self.logger.add_scalar("enu position/y", enu_pos[1], time*100)
        self.logger.add_scalar("enu position/z", enu_pos[2], time*100)

        enu_next_pos = pm.ecef2enu(*(self.filter.x[:3] + self.filter.x[3:6]), *self.environment.get_cam_pos_gps())
        enu_vel = np.array(enu_next_pos) - np.array(enu_pos)

        self.logger.add_scalar("enu velocity/x", enu_vel[0], time*100)
        self.logger.add_scalar("enu velocity/y", enu_vel[1], time*100)
        self.logger.add_scalar("enu velocity/z", enu_vel[2], time*100)

        self.logger.add_scalar("bearing/azimuth", target_az, time*100)
        self.logger.add_scalar("bearing/altitude", target_alt, time*100)

        az_err = current_az - target_az
        alt_err = current_alt - target_alt

        self.logger.add_scalar("mount/actual_azimuth", current_az, time*100)
        self.logger.add_scalar("mount/actual_altitude", current_alt, time*100)

        if not control_scope:
            return 

        distance_to_rocket = np.linalg.norm(enu_pos)
        focuser_pos = self.dof_calc.get_focuser_offset_for_object(distance_to_rocket)
        focuser_pos = np.clip(focuser_pos, *self.environment.get_focuser_bounds())
        self.logger.add_scalar("mount/focuser_pos", focuser_pos, time*100)
        self.logger.add_scalar("mount/distance", distance_to_rocket, time*100)
        self.environment.move_focuser(focuser_pos)

        # input_x = self.x_controller.step(-az_err)
        # input_y = self.y_controller.step(-alt_err)
        input_x, input_y = self.mpc_controller.step(self.filter, (current_az, current_alt))
        MAX_SLEW_RATE_AZI = 8 
        MAX_SLEW_RATE_ALT = 6
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        min_y_input = 0 if current_alt <=0 else -MAX_SLEW_RATE_ALT
        y_clipped = np.clip(input_y,min_y_input,MAX_SLEW_RATE_ALT)
        self.environment.move_telescope(x_clipped, y_clipped)
        self.logger.add_scalar("mount/x_input", x_clipped, time*100)
        self.logger.add_scalar("mount/y_input", y_clipped, time*100)

    def stop_tracking(self):
        self.x_controller = PIDController(5,1,1)
        self.y_controller = PIDController(5,1,1)
        self.active_tracking = False
