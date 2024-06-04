import pygame
from telescope import Telescope
from zwo_asi import ASICamera
from tracker import Tracker
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from time import time, strftime

from zwo_eaf import EAF, getNumEAFs, getEAFID

# outside rocket lab garage
MOUNT_GPS_LAT = 33.6449307
MOUNT_GPS_LNG = -117.8413249

t = Telescope()
# reasonable range for gain is 50-1000
gain = 150
exposure = 1/30
cam = ASICamera(0,gain,exposure)
focal_length_pixels = 12.5 / 2.9e-3
logger = SummaryWriter(f'runs/telescope_control')
tracker = Tracker((1920, 1080), focal_length_pixels, logger, t, t.read_position(), 100, (MOUNT_GPS_LAT,MOUNT_GPS_LNG))

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
    global recording, video, gain, tracking, cam
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
        cam = ASICamera(0,gain,exposure)
    
    elif  button == BUTTON_RIGHT_TRIGGER:
        gain -= 50 
        cam = ASICamera(0,gain,exposure)

    elif button == BUTTON_SQUARE:
        print("Tracking toggled")
        tracking = not tracking


def tracker_control(img):
    return tracker.update_tracking(img, None, 0, None)

start_time_tenth_second = (time()%1) 
pixel_pos = None

sun_azi_alt = (0, 90)

def main():
    global pixel_pos
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
        if event.type == pygame.JOYBUTTONDOWN:
            handle_button_press(event.button) 


    print("??")
    img = cam.get_frame()
    
    if tracking:
        ret_val = tracker_control(img)
        if ret_val is not None:
            pixel_pos = ret_val
    else:
        joystick_control()
    

    # # draw current position
    cv.rectangle(img, (10,50), (200,0), (0,0,0), -1)
    azi, alt = t.read_position()
    readout_text = ""
    readout_text += f"Gain: {gain}"
    readout_text += f"\nazi: {azi:.2f} alt: {alt:.2f}"
    for i, line in enumerate(readout_text.split('\n')):
        cv.putText(img, line, (10,20+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    

    if recording:
        video.write(img)


    # resize to half size
    scale_factor = 1
    img = cv.resize(img, (1920//scale_factor, 1080//scale_factor))

    # # draw a crosshair
    h,w = img.shape[:2]
    cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
    cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)

    circle_thickness = -1 if recording else 2
    cv.circle(img, (w-30,30),20,(0,0,255),circle_thickness)

    rect_thickness = -1 if tracking else 2
    cv.rectangle(img, (w-50, h-50), (w-20, h-20), (255,0,255), rect_thickness)

    if tracking:
        # draw green circle around pixel pos
        if ret_val is None:
            color = (0,0,255)
        else:
            color = (0,255,0)
        
        if pixel_pos is not None:
            cv.circle(img, pixel_pos//scale_factor, 10, color, 3)

    cv.imshow("Camera Image", img)

    # if press q, break

    if cv.waitKey(1) == ord('q'):
        t.stop()
        return False
    
    return True

if __name__ == '__main__':
    eaf_count = getNumEAFs()

    print(f"Found {eaf_count} EAFs")

    if eaf_count < 1:
        print("No EAFs found")
        exit()

    eaf_id = getEAFID(0)

    eaf = EAF(eaf_id)

    # main loop
    while True:
        try:
            if not main():
                break
        except Exception as e:
            t.stop()
            print(e)
            break