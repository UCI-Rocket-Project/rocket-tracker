from pid_controller import PIDController

from alpaca.telescope import Telescope, TelescopeAxes
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from filterpy.kalman import KalmanFilter
import cv2 as cv

class Tracker:
    def __init__(self, camera_res: tuple[int,int], logger: SummaryWriter, telescope: Telescope):
        self.camera_res = camera_res
        self.logger = logger
        self.x_controller = PIDController(0.015,0,0.01)
        self.y_controller = PIDController(0.015,0,0.01)
        # self.filter = KalmanFilter()
        self.telescope = telescope
        self.tracker = cv.TrackerNano()
        self.prev_img = np.zeros(self.camera_res)
        self.feature_detector = cv.ORB_create()
        self.target_feature: np.ndarray = None # feature description 
        self.SCALE_FACTOR = 2

    def update_image_tracking(self, img: np.ndarray) -> tuple[int,int]:
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, np.array(img.shape)[:2]//self.SCALE_FACTOR) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(img,None)

        if len(keypoints) == 0:
            return

        center = np.array(gray.shape)//2
        if self.target_feature is None:
            closest_dist = np.linalg.norm(center*2)
            for keypoint, description in zip(keypoints,descriptions):
                p = np.array(keypoint.pt)
                dist = np.linalg.norm(center-p)
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

        vis = cv.drawKeypoints(gray,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        box_size = np.array([20,20])
        cv.rectangle(gray, new_pos-box_size//2, new_pos+box_size//2, (0,255,0),2)
        cv.imwrite("features.png",vis)
        return new_pos*self.SCALE_FACTOR


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