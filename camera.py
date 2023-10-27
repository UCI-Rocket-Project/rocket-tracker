from alpaca.camera import Camera
from time import sleep
import numpy as np
import cv2 as cv

C = Camera('localhost:49669',0, protocol="")
C.Connected = True
# 28621 different error? (websockets request was expected)
C.BinX = 1
C.BinY = 1
# print(C.MaxBinX)
# C.Action(Camera.SupportedActions[])
C.StartExposure(0.02,False)
while not C.ImageReady:
    sleep(0.01)

img = np.array(C.ImageArray)
print(img.shape)
cv.imshow("Preview", img)
cv.waitKey(0)