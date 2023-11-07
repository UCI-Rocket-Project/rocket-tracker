from pid_controller import PIDController

from telescope import Telescope
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv

class Tracker:
    def __init__(self, camera_res: tuple[int,int], focal_len: int, logger: SummaryWriter, telescope: Telescope):
        '''
        `camera_res`: camera resolution (w,h) in pixels
        `focal_len`: focal length in pixels
        '''
        self.camera_res = camera_res
        self.focal_len = focal_len
        self.logger = logger
        self.x_controller = PIDController(10,0,1)
        self.y_controller = PIDController(10,0,1)
        # self.filter = KalmanFilter()
        self.telescope = telescope
        self.feature_detector = cv.SIFT_create()
        self.target_feature: np.ndarray = None # feature description 
        self.SCALE_FACTOR = 4

    def estimate_az_alt_from_img(self, img: np.ndarray, global_step: int, gt_pos: tuple[int,int]) -> tuple[float,float]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)

        returns (alt,az)
        '''
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, np.array(img.shape)[:2]//self.SCALE_FACTOR) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(gray,None)
        points = np.array([kp.pt for kp in keypoints])

        if len(keypoints) == 0:
            return None

        center = np.array(gray.shape)//2
        if self.target_feature is None:
            closest_dist = np.linalg.norm(center*2)
            for dist, description in zip(np.linalg.norm(points-center,axis=1),descriptions):
                if dist<closest_dist:
                    self.target_feature = description
                    closest_dist = dist
        
        # find coordinates of closest feature point
        max_similarity = 0
        new_pos = None
        new_feature = None
        similarities = descriptions @ self.target_feature
        for i, (keypoint, similarity) in enumerate(zip(keypoints,similarities)):
            if similarity > max_similarity:
                max_similarity = similarity
                new_pos = np.array(keypoint.pt).astype(int)
                new_feature = descriptions[i]

        if new_pos is None:
            return None

        altitude_from_image_processing = self.telescope.Altitude+np.rad2deg(np.arctan((center[1]-new_pos[1])*self.SCALE_FACTOR/self.focal_len))
        azimuth_from_image_processing = self.telescope.Azimuth+np.rad2deg(np.arctan((center[0]-new_pos[0])*self.SCALE_FACTOR/self.focal_len))
        
        self.target_feature = new_feature
        self.logger.add_scalar("Pixel Estimate Error (X)",new_pos[0]*self.SCALE_FACTOR-gt_pos[0],global_step)
        self.logger.add_scalar("Pixel Estimate Error (Y)",new_pos[1]*self.SCALE_FACTOR-gt_pos[1],global_step)

        self.logger.add_scalar("Altitude Estimate (Degrees)",altitude_from_image_processing,global_step)
        self.logger.add_scalar("Azimuth Estimate (Degrees)",azimuth_from_image_processing,global_step)

        return altitude_from_image_processing, azimuth_from_image_processing


    def update_tracking(self, img: np.ndarray, global_step: int, gt_pos: tuple[int,int], pos_estimate: np.ndarray) -> tuple[int,int]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)
        `pos_estimate`: estimated position of rocket relative to the mount, where the mount 
        is at (0,0,0) and (0,0) az/alt is  towards positive Y, and Z is up
        '''

        altitude_from_image_processing, azimuth_from_image_processing = self.estimate_az_alt_from_img(img, global_step, gt_pos) or (None,None)

        altitude_from_pos_estimate = np.rad2deg(np.arctan2(pos_estimate[2], np.sqrt(pos_estimate[0]**2 + pos_estimate[1]**2)))
        azimuth_from_pos_estimate = -np.rad2deg(np.arctan2(pos_estimate[0], pos_estimate[1]))

        if altitude_from_image_processing is not None:
            alt_setpoint = altitude_from_image_processing
            az_setpoint = azimuth_from_image_processing
        else:
            alt_setpoint = altitude_from_pos_estimate
            az_setpoint = azimuth_from_pos_estimate

        # vis = cv.drawKeypoints(gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # box_size = np.array([20,20])
        # cv.rectangle(gray, new_pos-box_size//2, new_pos+box_size//2, (0,255,0),2)
        # cv.imwrite("features.png",vis)
        
        alt_err = alt_setpoint-self.telescope.Altitude
        az_err = az_setpoint-self.telescope.Azimuth

        self.logger.add_scalar("Altitude Tracking Error (Degrees)",alt_err,global_step)
        self.logger.add_scalar("Azimuth Tracking Error (Degrees)",az_err,global_step)

        input_x = self.x_controller.step(az_err)
        input_y = self.y_controller.step(alt_err)
        x_clipped = np.clip(input_x,-6,6)
        y_clipped = np.clip(input_y,-6,6)
        self.logger.add_scalar("X Input", x_clipped, global_step)
        self.logger.add_scalar("Y Input", y_clipped, global_step)
        self.telescope.slewAltitudeRate(y_clipped, global_step/100)
        self.telescope.slewAzimuthRate(x_clipped, global_step/100)