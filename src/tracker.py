from .pid_controller import PIDController

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from .utils import GroundTruthTrackingData, TelemetryData
from pymap3d import enu2geodetic
from dataclasses import dataclass
from .environment import Environment
from .rocket_filter import RocketFilter

@dataclass
class Feature:
    descriptor: np.ndarray
    size: float

class Tracker:
    def __init__(self, 
                environment: Environment,
                logger: SummaryWriter, 
                ):
        '''
        '''

        self.camera_res = environment.get_camera_resolution()
        self.focal_len = environment.get_focal_length()
        self.logger = logger
        self.x_controller = PIDController(10,0,1)
        self.y_controller = PIDController(10,0,1)
        self.gps_pos = environment.get_cam_pos_gps() # initial position of mount in GPS coordinates (lat,lng,alt)
        self.environment = environment

        self.filter = RocketFilter(environment.get_pad_pos_gps(), environment.get_cam_pos_gps())

        self.feature_detector = cv.SIFT_create()
        self.target_feature: np.ndarray = None # feature description 
        self.initial_feature_size = None
        self.SCALE_FACTOR = 4

        self.focuser_zero_distance = 300 # This is the total length of spacers we're using between the scope and camera, in mm
        self.focuser_focal_length = 714


    def _rocket_state_transition(self, state: np.ndarray, dt):
        '''
        x is x,dx,d2x,y,dy,d2y,z,dz,d2z
        '''

        x,y,z = state[0],state[3],state[6]
        dx,dy,dz = state[1],state[4],state[7]
        d2x,d2y,d2z = state[2],state[5],state[8]

        return np.array([
            x+dx*dt+0.5*d2x*dt**2,
            dx+d2x*dt,
            d2x,
            y+dy*dt+0.5*d2y*dt**2,
            dy+d2y*dt,
            d2y,
            z+dz*dt+0.5*d2z*dt**2,
            dz+d2z*dt,
            d2z
        ])

    def estimate_az_alt_scale_from_img(self, img: np.ndarray, global_step: int, gt_pos: tuple[int,int]) -> tuple[float,float,float]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)

        returns (alt,az,scale) where alt and az are in degrees and scale has no units, it's just the current apparent object size
        divided by the initial apparent object size
        '''
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        h,w = img.shape[:2]
        gray = cv.resize(gray, np.array([w,h])//self.SCALE_FACTOR) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(gray,None)
        # visualize keypoints on image
        # cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("Keypoints",gray)
        points = np.array([kp.pt for kp in keypoints])

        if len(keypoints) == 0:
            return None, None, None, None

        center = np.array(gray.shape)//2
        if self.target_feature is None:
            closest_dist = np.linalg.norm(center)
            for dist, description, keypoint in zip(np.linalg.norm(points-center,axis=1),descriptions, keypoints):
                if dist<closest_dist:
                    self.target_feature = Feature(description, keypoint.size)
                    self.initial_feature_size = keypoint.size
                    closest_dist = dist
        # find coordinates of closest feature point
        max_similarity = 0
        new_pos = None
        new_feature = None
        similarities = descriptions @ self.target_feature.descriptor
        for i, (keypoint, similarity) in enumerate(zip(keypoints,similarities)):
            if similarity > max_similarity:
                max_similarity = similarity
                new_pos = np.array(keypoint.pt).astype(int)
                new_feature = Feature(descriptions[i], keypoint.size)
        self.target_feature = new_feature

        if new_pos is None:
            return None, None, None, None

        t_azi, t_alt = self.environment.get_telescope_orientation()

        altitude_from_image_processing = t_alt+np.rad2deg(np.arctan((center[1]-new_pos[1])*self.SCALE_FACTOR/self.focal_len))
        azimuth_from_image_processing = t_azi+np.rad2deg(np.arctan((center[0]-new_pos[0])*self.SCALE_FACTOR/self.focal_len))
        
        if gt_pos is not None:
            self.logger.add_scalar("Pixel Estimate Error (X)",new_pos[0]*self.SCALE_FACTOR-gt_pos[0],global_step)
            self.logger.add_scalar("Pixel Estimate Error (Y)",new_pos[1]*self.SCALE_FACTOR-gt_pos[1],global_step)

        self.logger.add_scalar("Imaging Altitude Estimate (Degrees)",altitude_from_image_processing,global_step)
        self.logger.add_scalar("Imaging Azimuth Estimate (Degrees)",azimuth_from_image_processing,global_step)

        pixel_loc = np.array(new_pos) * self.SCALE_FACTOR

        return altitude_from_image_processing, azimuth_from_image_processing, new_feature.size/self.initial_feature_size, pixel_loc


    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData, global_step: int, ground_truth: GroundTruthTrackingData) -> tuple[int,int]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        altitude_from_image_processing, azimuth_from_image_processing, img_scale, pixel_pos = self.estimate_az_alt_scale_from_img(img, global_step, ground_truth.pixel_coordinates if ground_truth else None)

        using_image_processing = altitude_from_image_processing is not None

        self.logger.add_scalar("Using image processing", int(using_image_processing), global_step)

        filter_pred_time = global_step-self.previous_filter_predict_time
        self.filter.predict(filter_pred_time/100)
        self.previous_filter_predict_time = global_step

        # TODO: set measurement noise really high for any missing measurements
        np.set_printoptions(suppress=True, precision=5)
        # if using_image_processing:
        if self.use_telem:
            measurement_vector = np.array([
                altitude_from_image_processing,
                azimuth_from_image_processing, 
                # img_scale,
                telem_measurements.gps_lat,
                telem_measurements.gps_lng,
                telem_measurements.altimeter_reading,
                telem_measurements.accel_x,
                telem_measurements.accel_y,
                telem_measurements.accel_z,
            ])
        else:
            measurement_vector = np.array([
                altitude_from_image_processing,
                azimuth_from_image_processing, 
            ])

        # print(measurement_vector)
        # print(self._measurement_function(self.filter.x))
        # print()
        if using_image_processing: # TODO make this actually conditional
            # missing_measurements = np.array([m is None for m in measurement_vector])
            # measurement_covariance = self.filter.R
            # measurement_covariance[missing_measurements,missing_measurements] = 1e9
            # measurement_vector[missing_measurements] = 0
            self.filter.update(measurement_vector)
                            
        self.logger.add_scalar("Rocket Position X", self.filter.x[0], global_step)
        self.logger.add_scalar("Rocket Position Y", self.filter.x[3], global_step)
        self.logger.add_scalar("Rocket Position Z", self.filter.x[6], global_step)

        self.logger.add_scalar("Rocket Velocity X", self.filter.x[1], global_step)
        self.logger.add_scalar("Rocket Velocity Y", self.filter.x[4], global_step)
        self.logger.add_scalar("Rocket Velocity Z", self.filter.x[7], global_step)

        self.logger.add_scalar("Rocket Acceleration X", self.filter.x[2], global_step)
        self.logger.add_scalar("Rocket Acceleration Y", self.filter.x[5], global_step)
        self.logger.add_scalar("Rocket Acceleration Z", self.filter.x[8], global_step)
        # vis = cv.drawKeypoints(gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # box_size = np.array([20,20])
        # cv.rectangle(gray, new_pos-box_size//2, new_pos+box_size//2, (0,255,0),2)
        # cv.imwrite("features.png",vis)
        
        if using_image_processing:
            alt_setpoint, az_setpoint, *_ = measurement_vector
        else:
            alt_setpoint, az_setpoint, *_ = self._measurement_function(self.filter.x)
        # az_setpoint, alt_setpoint = ground_truth.az_alt 

        # estimate distance to target
        cur_state = self.filter.x
        dist = np.linalg.norm(cur_state[:3])
        focuser_position = self.focuser_focal_length*(1/self.focal_len)*dist+self.focuser_zero_distance
        self.focuser.move_to(focuser_position)

        t_azi, t_alt = self.telescope.read_position()

        alt_err = alt_setpoint-t_alt
        az_err = az_setpoint-t_azi

        self.logger.add_scalar("Altitude Tracking Error (Degrees)",alt_err,global_step)
        self.logger.add_scalar("Azimuth Tracking Error (Degrees)",az_err,global_step)

        input_x = self.x_controller.step(az_err)
        input_y = self.y_controller.step(alt_err)
        MAX_SLEW_RATE_AZI = 8
        MAX_SLEW_RATE_ALT = 6
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE_AZI,MAX_SLEW_RATE_AZI)
        y_clipped = np.clip(input_y,-MAX_SLEW_RATE_ALT,MAX_SLEW_RATE_ALT)
        self.logger.add_scalar("X Input", x_clipped, global_step)
        self.logger.add_scalar("Y Input", y_clipped, global_step)
        if self.telescope.is_fake:
            self.telescope.slew_rate_azi_alt(x_clipped, y_clipped, global_step/100)
        else:
            self.telescope.slew_rate_azi_alt(x_clipped, y_clipped)

        return pixel_pos