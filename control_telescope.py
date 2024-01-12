import pygame
from telescope import Telescope
from camera import Camera
from simulation.tracker import Tracker
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from time import time, strftime

t = Telescope()
gain = 300
cam = Camera(gain=gain)
focal_length_pixels = 12.5 / 2.9e-3
logger = SummaryWriter(f'runs/telescope_control')
tracker = Tracker((1920, 1080), focal_length_pixels, logger, t, (0,0,0), (0,0,-100))

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

BUTTON_X = 0
BUTTON_CIRCLE = 1
BUTTON_TRIANGLE = 2
BUTTON_SQUARE = 3

BUTTON_LEFT_BUMPER = 4
BUTTON_RIGHT_BUMPER = 5

BUTTON_LEFT_TRIGGER = 6
BUTTON_RIGHT_TRIGGER = 7

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

recording = False
video = None
tracking = False

def handle_button_press(button: int):
    global recording, video, gain, tracking
    if button == BUTTON_CIRCLE:
        if not recording:
            # start recording
            video = cv.VideoWriter(f"video_{strftime('%m_%m_%d-%H:%M')}.avi", cv.VideoWriter_fourcc(*'MJPG'), 10, (1920,1080))
        else:
            # stop recording
            video.release()
            video = None
        recording = not recording
    
    elif button == BUTTON_RIGHT_BUMPER:
        gain += 100
        cam.set_gain(gain)
    
    elif  button == BUTTON_RIGHT_TRIGGER:
        gain -= 50 
        cam.set_gain(gain)

    elif button == BUTTON_TRIANGLE:
        print("Tracking toggled")
        tracking = not tracking


def tracker_control(img):
    tracker.update_tracking(img, None, 0, None)

start_time_tenth_second = (time()%1) 

def main():
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
        if event.type == pygame.JOYBUTTONDOWN:
            handle_button_press(event.button) 


    img = cam.take_picture()
    if tracking and False:
        tracker_control(img)
    else:
        joystick_control()

    # # draw current position
    cv.rectangle(img, (10,50), (200,0), (0,0,0), -1)
    # azi, alt = t.read_position()
    readout_text = ""
    readout_text += f"Gain: {gain}"
    # readout_text += f"\nazi: {azi:.2f} alt: {alt:.2f}"
    cv.putText(img, readout_text, (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    

    if recording:
        video.write(img)


    # resize to half size
    img = cv.resize(img, (960,540))

    # # draw a crosshair
    h,w = img.shape[:2]
    cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
    cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)
    if recording:
        cv.circle(img, (w-30,30),20,(0,0,255),-1)

    if tracking:
        cv.rectangle(img, (w-50, h-50), (w-20, h-20), (255,0,0), -1)

    cv.imshow("Camera Image", img)

    # if press q, break

    if cv.waitKey(1) == ord('q'):
        t.stop()
        return False
    
    return True


# main loop
while True:
    try:
        if not main():
            break
    except Exception as e:
        t.stop()
        print(e)
        break
