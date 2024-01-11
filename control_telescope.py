import pygame
from telescope import Telescope
from camera import Camera
# from simulation.tracker import Tracker
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from time import time

t = Telescope()
cam = Camera()
focal_length_pixels = 12.5 / 2.9e-3
logger = SummaryWriter(f'runs/telescope_control')
# tracker = Tracker((1920, 1080), focal_length_pixels, logger, t, (0,0,0), (0,0,-100))

pygame.init()
pygame.joystick.init()
print(f"Joysticks: {pygame.joystick.get_count()}")
controller = pygame.joystick.Joystick(0)
controller.init()

axis = {}

LEFT_AXIS_X = 0
LEFT_AXIS_Y = 1
RIGHT_AXIS_X = 3
RIGHT_AXIS_Y = 4

def joystick_control():
    
    slew_x = 0
    slew_y = 0
    
    if LEFT_AXIS_X in axis and abs(axis[LEFT_AXIS_X])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x = clamped*8
    
    if LEFT_AXIS_Y in axis and abs(axis[LEFT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_Y]) # use tanh to clamp to [-1,1]
        slew_y = clamped*6

    # duplicate for right axis
    if RIGHT_AXIS_X in axis and abs(axis[RIGHT_AXIS_X])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x += clamped
    
    if RIGHT_AXIS_Y in axis and abs(axis[RIGHT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[RIGHT_AXIS_Y])
        slew_y += clamped

    t.slew_rate_azi_alt(slew_x, slew_y)

# def tracker_control(img):
#     tracker.update_tracking(img, None, 0, None)

start_time_tenth_second = (time()%1) 
# main loop
while True:
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
    joystick_control()

    img = cam.take_picture()

    # draw a crosshair
    h,w = img.shape[:2]
    cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
    cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)

    # draw current position
    azi, alt = t.read_position()
    cv.rectangle(img, (10,20), (200,0), (0,0,0), -1)
    cv.putText(img, f"azi: {azi:.2f} alt: {alt:.2f}", (10,10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    cv.imshow("Camera Image", img)

    # if press q, break

    if cv.waitKey(1) == ord('q'):
        break 

