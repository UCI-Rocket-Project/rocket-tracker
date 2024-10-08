import line_profiler
from src.component_algos.pid_controller import PIDController
from src.component_algos.mpc_controller import MPCController
from src.component_algos.launch_detector import LaunchDetector
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.utils import TelemetryData
from src.environment import Environment
from src.component_algos.rocket_filter import RocketFilter
from src.component_algos.img_tracking import YOLOImageTracker, NoDetectionError, BaseImageTracker
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
import cv2 as cv
from src.component_algos.depth_of_field import DOFCalculator, MM_PER_PIXEL
import traceback


class Tracker:
    def __init__(self, 
                environment: Environment,
                logger: SummaryWriter, 
                img_tracker: BaseImageTracker
                ):
        '''
        img_tracker_class needs to extend BaseImageTracker
        '''

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

        self.img_tracker = img_tracker
        self.active_tracking = False

    def _pixel_pos_to_az_alt(self, pixel_pos: np.ndarray) -> tuple[float,float]:
        az = -np.arctan2(pixel_pos[0] - self.camera_res[0] / 2, self.focal_len_pixels)
        alt = -np.arctan2(pixel_pos[1] - self.camera_res[1] / 2, self.focal_len_pixels)
        pixel_rot = R.from_euler("ZY", [az, alt], degrees=False)
        initial_rot = R.from_euler("ZY", self.environment.get_telescope_orientation(), degrees=True)

        final_rot = initial_rot * pixel_rot
        az, alt = final_rot.as_euler("ZYX", degrees=True)[:2]
        return az, alt

    def _az_alt_to_pixel_pos(self, az_alt: np.ndarray) -> tuple[float,float]:
        telescope_current_rotation = R.from_euler("ZY", self.environment.get_telescope_orientation(), degrees=True)
        az_alt_vector = R.from_euler("ZY", az_alt, degrees=True).apply([0,0,1])
        cam_vec = telescope_current_rotation.inv().apply(az_alt_vector) 
        pixel_x = self.camera_res[0]/2 + self.focal_len_pixels * cam_vec[0]/cam_vec[2]
        pixel_y = self.camera_res[1]/2 + self.focal_len_pixels * cam_vec[1]/cam_vec[2]
        return int(pixel_x), int(pixel_y)

    def _ecef_to_az_alt(self, ecef_pos: np.ndarray) -> tuple[float,float]:
        return self.filter.hx_bearing(ecef_pos)[:2]
    
    def start_tracking(self, initial_cam_orientation: tuple[float,float]):
        self.initial_cam_orientation = initial_cam_orientation
        self.filter = RocketFilter(self.environment.get_pad_pos_gps(), self.gps_pos, self.initial_cam_orientation, self.focal_len_pixels, writer=self.logger)
        self.launch_detector = None
        self.active_tracking = True
        self.img_tracker.start_new_tracking()

    @line_profiler.profile
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
                bbox = self.img_tracker.estimate_pos(img)
                pixel_pos = bbox[:2]
            except NoDetectionError:
                pixel_pos = None

        if not self.active_tracking:
            return
        
        if pixel_pos is not None:
            self.logger.add_scalar("pixel position/x", pixel_pos[0], time*100)
            self.logger.add_scalar("pixel position/y", pixel_pos[1], time*100)
            if self.launch_detector is None:
                self.launch_detector = LaunchDetector(pixel_pos)

            if not self.launch_detector.has_detected_launch():
                self.launch_detector.update(pixel_pos, time)
                if self.launch_detector.has_detected_launch():
                    self.filter.set_launch_time(self.launch_detector.get_launch_time())

        if self.launch_detector is None:
            raise RuntimeError("Need to see rocket in first frame to start tracking, no rocket found.")

        if not self.launch_detector.has_detected_launch():
            self.logger.add_scalar("launched", 0, time*100)
            return
        self.logger.add_scalar("launched", 1, time*100)

        # calculate azimuth and altitude based on pixel position

        if pixel_pos is not None:
            az, alt = self._pixel_pos_to_az_alt(pixel_pos)
            bbox_diagonal_len = np.linalg.norm(bbox[2:])
            self.filter.predict_update_bearing(time, np.array([az, alt, bbox_diagonal_len]))

        pred_measurement = self.filter.hx_bearing(self.filter.x)
        predicted_pixel_pos = self._az_alt_to_pixel_pos(pred_measurement[:2])
        cv.circle(img, predicted_pixel_pos, 10, (255,0,0), 2)
        self.logger.add_scalar("pixel position/x_filter", predicted_pixel_pos[0], time*100)
        self.logger.add_scalar("pixel position/y_filter", predicted_pixel_pos[1], time*100)
        self.logger.add_scalar("pixel position/bbox_diag", pred_measurement[2], time*100)

        if telem_measurements is not None:
            ecef_pos = pm.geodetic2ecef(telem_measurements.gps_lat, telem_measurements.gps_lng, telem_measurements.gps_height)
            ned_vel = np.array([telem_measurements.v_north, telem_measurements.v_east, telem_measurements.v_down])
            enu_vel = np.array([ned_vel[1], ned_vel[0], -ned_vel[2]])
            ecef_vel = np.array(pm.enu2ecef(*enu_vel, telem_measurements.gps_lat, telem_measurements.gps_lng, telem_measurements.gps_height)) - np.array(ecef_pos)

            z = np.array([*ecef_pos, *ecef_vel])
            self.logger.add_scalar("telemetry/lat", telem_measurements.gps_lat, time*100)
            self.logger.add_scalar("telemetry/lng", telem_measurements.gps_lng, time*100)
            self.logger.add_scalar("telemetry/gps_alt", telem_measurements.gps_height, time*100)
            self.logger.add_scalar("telemetry/v_north", telem_measurements.v_north, time*100)
            self.logger.add_scalar("telemetry/v_east", telem_measurements.v_east, time*100)
            self.logger.add_scalar("telemetry/v_down", telem_measurements.v_down, time*100)
            self.filter.predict_update_telem(time, z)
        
        if pixel_pos is None and telem_measurements is None:
            try:
                self.filter.predict(time)
            except np.linalg.LinAlgError:
                traceback.print_exc()
                print(self.filter.P)
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()

                ax.matshow(self.filter.P, cmap=plt.cm.Blues)
                plt.show()

        current_az, current_alt = self.environment.get_telescope_orientation()
        ecef_pos = self.filter.x[:3]
        enu_pos = pm.ecef2enu(*ecef_pos, *self.environment.get_pad_pos_gps())
        self.logger.add_scalar("enu position/x", enu_pos[0], time*100)
        self.logger.add_scalar("enu position/y", enu_pos[1], time*100)
        self.logger.add_scalar("enu position/z", enu_pos[2], time*100)

        enu_next_pos = pm.ecef2enu(*(self.filter.x[:3] + self.filter.x[3:6]), *self.environment.get_pad_pos_gps())
        enu_vel = np.array(enu_next_pos) - np.array(enu_pos)

        self.logger.add_scalar("enu velocity/x", enu_vel[0], time*100)
        self.logger.add_scalar("enu velocity/y", enu_vel[1], time*100)
        self.logger.add_scalar("enu velocity/z", enu_vel[2], time*100)

        target_az, target_alt = self._ecef_to_az_alt(ecef_pos)
        self.logger.add_scalar("bearing/azimuth", target_az, time*100)
        self.logger.add_scalar("bearing/altitude", target_alt, time*100)


        az_err = current_az - target_az
        alt_err = current_alt - target_alt

        # self.logger.add_scalar("mount/actual_azimuth", current_az, time*100)
        # self.logger.add_scalar("mount/actual_altitude", current_alt, time*100)
        self.logger.add_scalars("mount/azimuth", {
            'actual': current_az,
            'setpoint': target_az,
            'error': az_err
        }, time*100)
        self.logger.add_scalars("mount/altitude", {
            'actual': current_alt,
            'setpoint': target_alt,
            'error': alt_err
        }, time*100)

        if not control_scope:
            return 

        distance_to_rocket = np.linalg.norm(pm.ecef2enu(*ecef_pos, *self.environment.get_cam_pos_gps()))
        focuser_pos = self.dof_calc.get_focuser_offset_for_object(distance_to_rocket)
        focuser_pos = np.clip(focuser_pos, *self.environment.get_focuser_bounds())
        self.logger.add_scalar("mount/focuser_pos", focuser_pos, time*100)
        self.logger.add_scalar("mount/distance", distance_to_rocket, time*100)
        self.environment.move_focuser(focuser_pos)

        mpc_input = self.mpc_controller.step(self.filter, (current_az, current_alt))
        input_x = self.x_controller.step(-az_err)
        input_y = self.y_controller.step(-alt_err)
        MAX_SLEW_RATE_AZI = 8 
        MAX_SLEW_RATE_ALT = 6
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        min_y_input = 0 if current_alt <=0 else -MAX_SLEW_RATE_ALT
        y_clipped = np.clip(input_y,min_y_input,MAX_SLEW_RATE_ALT)
        self.environment.move_telescope(x_clipped, y_clipped)
        self.logger.add_scalar("mount/x_input", x_clipped, time*100)
        self.logger.add_scalar("mount/y_input", y_clipped, time*100)
        self.logger.add_scalar("mount/mpc_x_input", mpc_input[0], time*100)
        self.logger.add_scalar("mount/mpc_y_input", mpc_input[1], time*100)

    def stop_tracking(self):
        self.x_controller = PIDController(5,5,1)
        self.y_controller = PIDController(5,5,1)
        self.active_tracking = False
