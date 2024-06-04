import numpy as np
import cv2 as cv
from ultralytics import YOLO
from dataclasses import dataclass
import os

CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Feature:
    descriptor: np.ndarray
    size: float

class NoDetectionError(RuntimeError):
    pass

class ImageTracker:
    def __init__(self):
        self.model = YOLO(f"{CURRENT_FILEPATH}/rocket_yolov8n.pt")

    def estimate_pos(self, img: np.ndarray) -> tuple[int, int, int, int]:
        '''
        Returns bounding box [center_x, center_y, width, height] of the highest confidence detection
        '''

        yolo_results = self.model.predict(img, verbose=False)[0]

        if len(yolo_results) == 0:
            raise NoDetectionError("No YOLO detections found in image")

        # get highest confidence detection
        
        boxes = yolo_results.boxes.xywh
        conf = yolo_results.boxes.conf
        print(boxes, conf)

        max_conf_idx = np.argmax(conf)
        box = boxes[max_conf_idx].int().tolist()

        # draw box
        cv.rectangle(img, (box[0]-box[2]//2, box[1]-box[3]//2), (box[0] + box[2]//2, box[1] + box[3]//2), (0, 255, 0), 2)
        return tuple(box)
        