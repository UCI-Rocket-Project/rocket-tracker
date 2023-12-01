import camera_zwo_asi
import cv2 as cv
import numpy as np

class Camera:
    def __init__(self):
        self.camera = camera_zwo_asi.Camera(0)
        self.camera.set_control('Gain', 300)
        self.camera.set_control('Exposure', 'auto')

    def take_picture(self) -> cv.Mat:
        img: np.ndarray = self.camera.capture().get_image()
        img = cv.cvtColor(img, cv.COLOR_BAYER_RGGB2BGR)
        return img

if __name__ == '__main__':
    cam = Camera()
    img = cam.take_picture()
    cv.imshow("Camera Image", img)
    cv.waitKey(0)