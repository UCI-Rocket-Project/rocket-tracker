import camera_zwo_asi
import cv2 as cv
import numpy as np

class Camera:
    def __init__(self, gain=300, exposure='auto'):
        self.camera = camera_zwo_asi.Camera(0)
        self.camera.set_control('Gain', gain)
        self.camera.set_control('Exposure', exposure)

    def take_picture(self) -> cv.Mat:
        img: np.ndarray = self.camera.capture().get_image()
        img = cv.cvtColor(img, cv.COLOR_BAYER_RGGB2BGR)
        return img
    
    def set_gain(self, new_gain: int):
        self.camera.set_control('Gain', new_gain)

    def set_exposure(self, new_exposure: int):
        self.camera.set_control('Exposure', new_exposure)



if __name__ == '__main__':
    cam = Camera()
    img = cam.take_picture()
    cv.imshow("Camera Image", img)
    cv.waitKey(0)