import numpy as np
import cv2 as cv
from dataclasses import dataclass

@dataclass
class Feature:
    descriptor: np.ndarray
    size: float

class NoKeypointsFoundError(RuntimeError):
    pass

class ImageTracker:
    def __init__(self, scale_factor=4):
        '''
        `scale_factor` controls how much the image is downscaled before we run detection on it. Higher
        values will make this faster but less accurate.
        '''
        self.target_feature = None
        self.feature_detector = cv.SIFT_create()
        self.scale_factor = scale_factor 

    def estimate_pos(self, img: np.ndarray) -> tuple[int, int]:
        '''
        `img`: image from camera
        `global_step`: current time step (for logging)
        `gt_pos`: ground truth pixel position of rocket (for logging error and mocking out the pixel tracking algorithm)

        if debug visualization is enabled, this function will draw the keypoints on the image

        returns (alt,az,scale) where alt and az are in degrees and scale has no units, it's just the current apparent object size
        divided by the initial apparent object size

        raises RuntimeError if no keypoints are found in the image
        '''
        gray_resized = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray_resized = cv.resize(gray_resized, (gray_resized.shape[1], gray_resized.shape[0])//self.scale_factor) # resize to make computation faster
        keypoints, descriptions = self.feature_detector.detectAndCompute(gray_resized,None)
        if len(keypoints) == 0:
            raise NoKeypointsFoundError("No keypoints found in image")

        # visualize keypoints on image
        debug_vis_keypoints = [cv.KeyPoint(kp.pt[0]*self.scale_factor, kp.pt[1]*self.scale_factor, kp.size) for kp in keypoints]
        cv.drawKeypoints(img, debug_vis_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # no imshow because we are already displaying the image in the main loop (hehehe side effects)

        points = np.array([kp.pt for kp in keypoints])

        # if first time, set target feature to closest feature to center of image
        if self.target_feature is None:
            center = np.array(gray_resized.shape)//2
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
        pixel_loc = np.array(new_pos) * self.scale_factor

        return pixel_loc[1], pixel_loc[0]