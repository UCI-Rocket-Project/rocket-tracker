import numpy as np
import cv2 as cv
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput

import os

from src.component_algos.img_tracking.base_image_tracker import BaseImageTracker, NoDetectionError

CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))

class YOLOImageTracker(BaseImageTracker):
    def __init__(self, use_coco = False):
        if use_coco:
            weights_file = 'coco_yolo11n.onnx'
        else:
            weights_file = 'rocket_yolo11n.onnx'
        self.yolo_pipeline = Pipeline.create(
            task="yolov8",
            model_path=f"{CURRENT_FILEPATH}/{weights_file}",   # sparsezoo stub or path to local ONNX
        )

        self.tracked_id = None
        self.reset_tracking = False

    def start_new_tracking(self):
        self.reset_tracking = True


    def estimate_pos(self, img: np.ndarray) -> tuple[int, int, int, int]:
        '''
        Returns bounding box [center_x, center_y, width, height] of the highest confidence detection
        '''

        # TODO: add tracking on top of this. See previous file history for how the logic for handling tracking ids was done.
        # DeepSparse doesn't support tracking yet, so we'll have to implement it ourselves.

        yolo_results: YOLOOutput = self.yolo_pipeline(images=[img])#self.model.track(img, verbose=False, persist=True)[0]

        boxes = yolo_results.boxes[0]
        scores = yolo_results.scores[0]
        if len(boxes) == 0:
            raise NoDetectionError("No YOLO detections found in image")


        ret = None
        ret_idx = np.argmax(scores)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = np.array(boxes[i]).astype(int)
            x1 = np.clip(x1, 0, img.shape[1])
            y1 = np.clip(y1, 0, img.shape[0])
            x2 = np.clip(x2, 0, img.shape[1])
            y2 = np.clip(y2, 0, img.shape[0])

            box_xywh = ((x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1)
            color = (255,0,0) if i==ret_idx else (0,0,255)
            if i == ret_idx:
                ret = box_xywh
            cv.putText(img, f"{scores[i]:.2f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.rectangle(img, (x1,y1), (x2,y2), color, 2)
        if np.max(scores) < 0.4: # TODO: instead of doing this, let the kalman filter reject low-confidence boxes that don't match the predictions
            raise NoDetectionError("No high confidence detection found in image")
        return ret
