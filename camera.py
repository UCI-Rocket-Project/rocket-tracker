import camera_zwo_asi
import cv2 as cv
import numpy as np 

# connecting to the camera
# at index 0
camera = camera_zwo_asi.Camera(0)

# printing information in the
# terminal

# changing some controllables
# (supported arguments: the one that are
# indicated as 'writable' in the information
# printed above)
camera.set_control("Gain",300)
camera.set_control("Exposure","auto")

# taking the picture
# filepath and show are optional, if you do not
# want to save the image or display it
img: np.ndarray = camera.capture().get_image()
img = cv.cvtColor(img, cv.COLOR_BAYER_RGGB2BGR)
print(img.shape)
cv.imshow("img",img)
cv.waitKey(0)

# getting a flat numpy array with the image data