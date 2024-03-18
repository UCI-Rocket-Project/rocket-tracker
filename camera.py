import camera_zwo_asi
import cv2 as cv
import numpy as np
from time import time

class Camera:
    def __init__(self, gain=300, exposure='auto'):
        self.camera = camera_zwo_asi.Camera(0)
        self.camera.set_control('Gain', gain)
        self.camera.set_control('Exposure', exposure)
        self.blank_img = self.camera.get_roi().get_image()

    def take_picture(self) -> cv.Mat:
        cam_start = time()
        self.camera.capture(self.blank_img)
        print(f"Cap took {time()-cam_start}")
        img = cv.cvtColor(self.blank_img.get_image(), cv.COLOR_BAYER_RGGB2BGR)
        return img

    def fill_picture(self, img, rgb) -> None:
        cam_start = time()
        self.camera.capture(img)
        print(f"Cap took {time()-cam_start}")
        cv.cvtColor(img, cv.COLOR_BAYER_RGGB2BGR, img)
    
    def set_gain(self, new_gain: int):
        self.camera.set_control('Gain', new_gain)

    def set_exposure(self, new_exposure: int):
        self.camera.set_control('Exposure', new_exposure)



if __name__ == '__main__':
    cam = Camera()
    img = cam.take_picture()
    cv.imshow("Camera Image", img)
    cv.waitKey(0)