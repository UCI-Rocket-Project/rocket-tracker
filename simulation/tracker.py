from pid_controller import PIDController

from telescope import Telescope
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
from utils import GroundTruthTrackingData, TelemetryData, enu_to_gps
from dataclasses import dataclass

@dataclass
class Feature:
    descriptor: np.ndarray
    size: float

class Tracker:
    def __init__(self, 
                camera_res: tuple[int,int], 
                focal_len: int, 
                logger: SummaryWriter, 
                telescope: Telescope, 
                rocket_initial_position: np.ndarray,
                mount_initial_position: np.ndarray):
        '''
        `camera_res`: camera resolution (w,h) in pixels
        `focal_len`: focal length in pixels
        `logger`: tensorboard logger
        `telescope`: telescope object for controlling the telescope with ASCOM Alpaca
        `rocket_initial_position`: initial position of the rocket in GPS coordinates (lat,lng)
        `mount_initial_position`: initial position of the mount in GPS coordinates (lat,lng)
        '''

        self.camera_res = camera_res
        self.focal_len = focal_len
        self.logger = logger
        self.x_controller = PIDController(10,0,1)
        self.y_controller = PIDController(10,0,1)
        self.gps_pos = mount_initial_position # initial position of mount in GPS coordinates (lat,lng,alt)

        state_dimensionality = 9

        self.filter = UnscentedKalmanFilter(
            dim_x = state_dimensionality, # state dimension (xyz plus their 1st and 2nd derivatives)
            dim_z = 3, # observation dimension (altitude, azimuth, scale, lat,lng, altimeter reading),
            dt = 1/30,
            hx = self._measurement_function,
            fx = self._rocket_state_transition,
            points = JulierSigmaPoints(state_dimensionality)
        )

        self.rocket_initial_position = rocket_initial_position
        self.filter.x  = np.zeros(state_dimensionality)
        self.filter.x[0], self.filter.x[3], self.filter.x[6] = rocket_initial_position
        self.previous_filter_predict_time = 0

        self.telescope = telescope
        self.feature_detector = cv.SIFT_create()
        self.target_feature: np.ndarray = None # feature description 
        self.initial_feature_size = None
        self.SCALE_FACTOR = 4

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

    def _measurement_function(self, state: np.ndarray):
        x,y,z = state[0],state[3],state[6]
        alt = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
        az = -np.rad2deg(np.arctan2(x, y))
        lat, lng, height = enu_to_gps(np.array([x,y,z]), self.gps_pos)
        initial_dist = np.linalg.norm(self.rocket_initial_position)
        new_dist = np.linalg.norm([x,y,z])
        scale = new_dist/initial_dist
        
        return np.array([
            alt,
            az,
            # scale,
            # lat,
            # lng,
            height
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
        gray = cv.resize(gray, np.array(img.shape)[:2]//self.SCALE_FACTOR) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(gray,None)
        points = np.array([kp.pt for kp in keypoints])

        if len(keypoints) == 0:
            return None, None, None

        center = np.array(gray.shape)//2
        if self.target_feature is None:
            closest_dist = np.linalg.norm(center*2)
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
            return None, None, None

        altitude_from_image_processing = self.telescope.Altitude+np.rad2deg(np.arctan((center[1]-new_pos[1])*self.SCALE_FACTOR/self.focal_len))
        azimuth_from_image_processing = self.telescope.Azimuth+np.rad2deg(np.arctan((center[0]-new_pos[0])*self.SCALE_FACTOR/self.focal_len))
        
        self.logger.add_scalar("Pixel Estimate Error (X)",new_pos[0]*self.SCALE_FACTOR-gt_pos[0],global_step)
        self.logger.add_scalar("Pixel Estimate Error (Y)",new_pos[1]*self.SCALE_FACTOR-gt_pos[1],global_step)

        self.logger.add_scalar("Imaging Altitude Estimate (Degrees)",altitude_from_image_processing,global_step)
        self.logger.add_scalar("Imaging Azimuth Estimate (Degrees)",azimuth_from_image_processing,global_step)

        return altitude_from_image_processing, azimuth_from_image_processing, new_feature.size/self.initial_feature_size


    def update_tracking(self, img: np.ndarray, telem_measurements: TelemetryData, global_step: int, ground_truth: GroundTruthTrackingData) -> tuple[int,int]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        altitude_from_image_processing, azimuth_from_image_processing, img_scale = self.estimate_az_alt_scale_from_img(img, global_step, ground_truth.pixel_coordinates)

        using_image_processing = altitude_from_image_processing is not None

        self.logger.add_scalar("Using image processing", int(using_image_processing), global_step)


        self.filter.predict(global_step-self.previous_filter_predict_time)
        self.previous_filter_predict_time = global_step

        # TODO: set measurement noise really high for any missing measurements
        np.set_printoptions(suppress=True, precision=5)
        # if using_image_processing:
        measurement_vector = np.array([
            altitude_from_image_processing,
            azimuth_from_image_processing, 
            # img_scale,
            # telem_measurements.gps_lat,
            # telem_measurements.gps_lng,
            telem_measurements.altimeter_reading
        ])
        # print(measurement_vector)
        # print(self._measurement_function(self.filter.x))
        # print()
        if using_image_processing:
            # missing_measurements = np.array([m is None for m in measurement_vector])
            # measurement_covariance = self.filter.R
            # measurement_covariance[missing_measurements,missing_measurements] = 1e9
            # measurement_vector[missing_measurements] = 0
            self.filter.update(measurement_vector)
                            
        self.logger.add_scalar("Kalman Filter x", self.filter.x[0], global_step)
        self.logger.add_scalar("Kalman Filter y", self.filter.x[3], global_step)
        self.logger.add_scalar("Kalman Filter z", self.filter.x[6], global_step)

        # vis = cv.drawKeypoints(gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # box_size = np.array([20,20])
        # cv.rectangle(gray, new_pos-box_size//2, new_pos+box_size//2, (0,255,0),2)
        # cv.imwrite("features.png",vis)
        
        # if using_image_processing:
        #     alt_setpoint, az_setpoint, *_ = measurement_vector
        # else:
        #     alt_setpoint, az_setpoint, *_ = self._measurement_function(self.filter.x)
        az_setpoint, alt_setpoint = ground_truth.az_alt 

        alt_err = alt_setpoint-self.telescope.Altitude
        az_err = az_setpoint-self.telescope.Azimuth

        self.logger.add_scalar("Altitude Tracking Error (Degrees)",alt_err,global_step)
        self.logger.add_scalar("Azimuth Tracking Error (Degrees)",az_err,global_step)

        input_x = self.x_controller.step(az_err)
        input_y = self.y_controller.step(alt_err)
        MAX_SLEW_RATE = 8
        x_clipped = np.clip(input_x,-MAX_SLEW_RATE,MAX_SLEW_RATE)
        y_clipped = np.clip(input_y,-MAX_SLEW_RATE,MAX_SLEW_RATE)
        self.logger.add_scalar("X Input", x_clipped, global_step)
        self.logger.add_scalar("Y Input", y_clipped, global_step)
        self.telescope.slewAltitudeRate(y_clipped, global_step/100)
        self.telescope.slewAzimuthRate(x_clipped, global_step/100)