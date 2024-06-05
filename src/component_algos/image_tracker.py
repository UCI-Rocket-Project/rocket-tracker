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
        self.tracked_id = None
        self.reset_tracking = False

    def start_new_tracking(self):
        self.reset_tracking = True


    def estimate_pos(self, img: np.ndarray) -> tuple[int, int, int, int]:
        '''
        Returns bounding box [center_x, center_y, width, height] of the highest confidence detection
        '''

        yolo_results = self.model.track(img, verbose=False, persist=True)[0]

        if len(yolo_results) == 0:
            raise NoDetectionError("No YOLO detections found in image")

        if yolo_results.boxes.id is None:
            raise NoDetectionError("No ID found in YOLO detections")

        found_box = None
        found_id = None
        found_conf = 0
        for conf, cls, xyxy, id in zip(yolo_results.boxes.conf, yolo_results.boxes.cls, yolo_results.boxes.xyxy, yolo_results.boxes.id):
            x1, y1, x2, y2 = xyxy.int().tolist()
            cx = img.shape[1]//2
            cy = img.shape[0]//2
            if self.reset_tracking and self.tracked_id is None and x1<cx<x2 and y1<cy<y2:
                self.tracked_id = id
                self.reset_tracking = False
                
            if x2-x1 < 100 and conf >= found_conf:
                found_box = xyxy.int().tolist()
                found_id = id
                found_conf = conf
            color = (255,0,0) if self.tracked_id is None else (0,0,255) # blue if not tracking, red if tracking
            cv.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv.putText(img, f"ID: {id}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
            
        if found_box is None:
            raise NoDetectionError("No tracked ID found in YOLO detections")
        
        if self.tracked_id is None:
            raise NoDetectionError("Haven't set tracked ID yet")

        # draw box
        x1, y1, x2, y2 = found_box
        cv.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv.putText(img, f"ID: {found_id}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        box_xywh = ((x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1)
        return box_xywh
        