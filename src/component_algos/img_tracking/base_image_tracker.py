
import numpy as np
import cv2 as cv
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput
from abc import ABC

import os

CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))

class NoDetectionError(RuntimeError):
    pass

class BaseImageTracker(ABC):
    def __init__(self):
        pass

    def start_new_tracking(self):
        pass


    def estimate_pos(self, img: np.ndarray) -> tuple[int, int, int, int]:
        '''
        Returns bounding box [center_x, center_y, width, height] of the highest confidence detection
        '''

        pass