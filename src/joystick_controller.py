from .environment import Environment
from .tracker import Tracker
import pygame
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from time import strftime

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

class JoystickController:
    def __init__(self, environment: Environment):
        self.environment  = environment
        pygame.init()
        pygame.joystick.init()

        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.joystick_axes = {}

        self.recording = False
        self.video_writer = None
        self.tracking = False

        self.gain = 0
        self.exposure = 1/30

        self.tracker = Tracker(environment, SummaryWriter("runs/tracker"))
        self.latest_tracker_pos: np.ndarray = None

        
    def _handle_button_press(self,button: int):
        if button == BUTTON_CIRCLE:
            if not self.recording:
                # start recording
                self.video_writer = cv.VideoWriter(f"video_{strftime('%m_%m_%d-%H:%M')}.avi", cv.VideoWriter_fourcc(*'MJPG'), 10, (1920,1080))
            else:
                # stop recording
                self.video_writer.release()
                self.video_writer = None
            self.recording = not self.recording
        
        elif button == BUTTON_RIGHT_BUMPER:
            self.gain += 100
            self.environment.set_camera_settings(self.gain, self.exposure)
        
        elif  button == BUTTON_RIGHT_TRIGGER:
            self.gain -= 50 
            self.environment.set_camera_settings(self.gain, self.exposure)

        elif button == BUTTON_SQUARE:
            print("Tracking toggled")
            self.tracking = not self.tracking

        elif button == BUTTON_X:
            print("Quitting")
            quit()

    def _joystick_control(self):
        slew_x = 0
        slew_y = 0

        axis = self.joystick_axes
        
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
        
        self.environment.move_telescope(-slew_x, -slew_y)

    def _update_display(self, img: np.ndarray):
        # draw current position
        cv.rectangle(img, (10,50), (200,0), (0,0,0), -1)
        azi, alt = self.environment.get_telescope_orientation()
        readout_text = ""
        readout_text += f"Gain: {self.gain}"
        readout_text += f"\nazi: {azi:.2f} alt: {alt:.2f}"
        for i, line in enumerate(readout_text.split('\n')):
            cv.putText(img, line, (10,20+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if self.recording:
            self.video_writer.write(img)

        # make tallest dimension 1080
        scale_factor = max(1, max(img.shape[:2])//1080)
        h,w = img.shape[:2]
        img = cv.resize(img, (w*scale_factor, h*scale_factor))

        # draw a crosshair
        h,w = img.shape[:2]
        cv.line(img, (w//2-50,h//2), (w//2+50,h//2), (0,0,255), 1)
        cv.line(img, (w//2,h//2-50), (w//2,h//2+50), (0,0,255), 1)

        circle_thickness = -1 if self.recording else 2
        cv.circle(img, (w-30,30),20,(0,0,255),circle_thickness)

        rect_thickness = -1 if self.tracking else 2
        cv.rectangle(img, (w-50, h-50), (w-20, h-20), (255,0,255), rect_thickness)

        if self.tracking:
            # draw green circle around pixel pos
            color = (0,255,0)
            
            if self.latest_tracker_pos is not None:
                cv.circle(img, self.latest_tracker_pos//scale_factor, 10, color, 3)

        cv.imshow("Camera Image", img)
    
    def loop_callback(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.joystick_axes[event.axis] = event.value
            if event.type == pygame.JOYBUTTONDOWN:
                self._handle_button_press(event.button) 
        
        if self.tracking:
            tracker_estimation = self.tracker.update_tracking(
                self.environment.get_camera_image(),
                self.environment.get_telescope_orientation(),
                0,
                None
            )
            if tracker_estimation is not None:
                self.latest_tracker_pos = tracker_estimation
        else:
            self._joystick_control()
        
        img = self.environment.get_camera_image()
        self._update_display(img)

        if cv.waitKey(1) == ord('q'):
            self.environment.move_telescope(0,0)
            return False
        
        return True
            

