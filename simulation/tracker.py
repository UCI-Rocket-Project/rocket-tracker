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
        self.rocket_az_estimate = 0
        self.rocket_alt_estimate = 0

    def update_tracking(self, img: np.ndarray, global_step: int, gt_pos: tuple[int,int]) -> tuple[int,int]:
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, np.array(img.shape)[:2]//self.SCALE_FACTOR) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(gray,None)
        points = np.array([kp.pt for kp in keypoints])

        if len(keypoints) == 0:
            return

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
        similarities = descriptions @ self.target_feature
        for keypoint, similarity in zip(keypoints,similarities):
            if similarity > max_similarity:
                max_similarity = similarity
                new_pos = np.array(keypoint.pt).astype(int)

        # uncomment this line to use ground truth pixel position instead of the one provided by the pixel tracking algorithm
        # new_pos = np.array(gt_pos)/self.SCALE_FACTOR

        self.logger.add_scalar("Pixel Estimate Error (X)",new_pos[0]*self.SCALE_FACTOR-gt_pos[0],global_step)
        self.logger.add_scalar("Pixel Estimate Error (Y)",new_pos[1]*self.SCALE_FACTOR-gt_pos[1],global_step)

        self.rocket_alt_estimate = self.telescope.Altitude+np.rad2deg(np.arctan((center[1]-new_pos[1])*self.SCALE_FACTOR/self.focal_len))
        self.rocket_az_estimate = self.telescope.Azimuth+np.rad2deg(np.arctan((center[0]-new_pos[0])*self.SCALE_FACTOR/self.focal_len))

        self.logger.add_scalar("Altitude Estimate (Degrees)",self.rocket_alt_estimate,global_step)
        self.logger.add_scalar("Azimuth Estimate (Degrees)",self.rocket_az_estimate,global_step)

        # vis = cv.drawKeypoints(gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # box_size = np.array([20,20])
        # cv.rectangle(gray, new_pos-box_size//2, new_pos+box_size//2, (0,255,0),2)
        # cv.imwrite("features.png",vis)

        alt_setpoint = self.rocket_alt_estimate
        az_setpoint = self.rocket_az_estimate

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