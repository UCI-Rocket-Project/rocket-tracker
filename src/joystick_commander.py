from .environment import Environment
import warnings
from .tracker import Tracker
from .vision_only_tracker import VisionOnlyTracker
from src.component_algos.img_tracking import YOLOImageTracker
import pygame
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from time import strftime
from typing import Callable
import line_profiler

LEFT_AXIS_X = 0
LEFT_AXIS_Y = 1
RIGHT_AXIS_X = 3
RIGHT_AXIS_Y = 4

BUTTON_X = 1
BUTTON_CIRCLE = 2
BUTTON_TRIANGLE = 3
BUTTON_SQUARE = 0

BUTTON_LEFT_BUMPER = 4
BUTTON_RIGHT_BUMPER = 5

BUTTON_LEFT_TRIGGER = 6
BUTTON_RIGHT_TRIGGER = 7

class JoystickCommander:
    def __init__(self, environment: Environment, logger: SummaryWriter, vision_only = False, auto_track_time = None):
        self.environment  = environment
        pygame.init()
        pygame.joystick.init()

        try:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            self.has_joystick = True
        except pygame.error:
            print("No joystick detected")
            self.has_joystick = False
        self.joystick_axes = {}

        self.recording = False
        self.video_writer = None
        self._tracking = False

        self.gain = 0
        self.exposure = int(1000*1/60)

        focuser_bounds = np.array(environment.get_focuser_bounds())
        # self.focuser_offsets = np.clip(np.logspace(*np.log10(focuser_bounds+1e-6), 100), *focuser_bounds)
        self.focuser_offsets = np.linspace(*focuser_bounds, 100)
        self.focuser_index = np.argmin(np.abs(self.focuser_offsets - environment.get_focuser_position()))

        self.logger = logger
        self._vision_only = vision_only
        self.tracker = Tracker(self.environment, self.logger, YOLOImageTracker()) if not self._vision_only else VisionOnlyTracker(self.environment, self.logger)
        self.latest_tracker_pos: np.ndarray = None
        self._auto_track_time = auto_track_time
        self._has_auto_tracked = False # flag to prevent auto-tracking from happening more than once
        self._image_size: tuple[int,int] = None
        cv.namedWindow('Camera Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Camera Image', (1000, 562))

    def _toggle_tracking(self):
        if not self._tracking:
            self.tracker.start_tracking(self.environment.get_telescope_orientation())
        else:
            self.tracker.stop_tracking()
        self._tracking = not self._tracking

    @property
    def tracking(self):
        return self._tracking
    
    def _toggle_recording(self):
        if not self.recording:
            # start recording
            self.video_writer = cv.VideoWriter(f"video_{strftime('%m_%m_%d-%H:%M')}.mp4", cv.VideoWriter_fourcc(*'MP4V'), 10, self._image_size)
        else:
            # stop recording
            self.video_writer.release()
            print("done releasing video writer")
            self.video_writer = None
        self.recording = not self.recording
        
    def _handle_button_press(self,button: int):
        if button == BUTTON_CIRCLE:
            self._toggle_recording()
        
        elif button == BUTTON_RIGHT_BUMPER:
            self.gain += 100
            self.environment.set_camera_settings(self.gain, self.exposure)
        
        elif  button == BUTTON_RIGHT_TRIGGER:
            self.gain -= 50 
            self.environment.set_camera_settings(self.gain, self.exposure)

        elif button == BUTTON_LEFT_BUMPER:
            new_index = min(self.focuser_index+2, len(self.focuser_offsets)-1)
            try:
                self.environment.move_focuser(int(self.focuser_offsets[self.focuser_index]))
                self.focuser_index = new_index
            except ValueError:
                warnings.warn('Ignored focuser command. It is currently in motion.')
        
        elif  button == BUTTON_LEFT_TRIGGER:
            new_index = max(self.focuser_index-1, 0)
            try:
                self.environment.move_focuser(int(self.focuser_offsets[self.focuser_index]))
                self.focuser_index = new_index
            except ValueError:
                warnings.warn('Ignored focuser command. It is currently in motion.')

        elif button == BUTTON_SQUARE:
            print("Tracking toggled")
            self._toggle_tracking()

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

    def _handle_keypress(self, key: int):
        if key == ord('q'):
            self.environment.move_telescope(0,0)
            quit()
        elif key == ord('w'):
            self.environment.move_telescope(0,1)
        elif key == ord('s'):
            self.environment.move_telescope(0,-1)
        elif key == ord('a'):
            self.environment.move_telescope(1,0)
        elif key == ord('d'):
            self.environment.move_telescope(-1,0)
        elif key == ord('t'):
            self._toggle_tracking()
        elif key == ord('r'):
            self._toggle_recording()
        elif not self.tracking and not self.has_joystick:
            self.environment.move_telescope(0,0)

    def _update_display(self, img: np.ndarray, time: float):
        # draw current position
        cv.rectangle(img, (10,100), (200,0), (0,0,0), -1)
        azi, alt = self.environment.get_telescope_orientation()
        readout_text = ""
        readout_text += f"Gain: {self.gain}\n"
        readout_text += f"Exposure: {self.exposure}\n"
        readout_text += f"Focuser: {self.focuser_offsets[self.focuser_index]:.7f}\n"
        readout_text += f"azi: {azi:.2f} alt: {alt:.2f}\n"
        readout_text += f"time: {time:.2f}\n"
        for i, line in enumerate(readout_text.split('\n')):
            cv.putText(img, line, (10,20+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


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

        if self.recording:
            self.video_writer.write(img)
        cv.imshow("Camera Image", img)

    @line_profiler.profile
    def loop_callback(self, time: float, img_debug_callback: Callable = lambda x: None) -> bool:
        '''
        img_debug_callback should take a cv Mat and draw on it as a side effect, then return nothing.
        The default value is a function that takes 1 argument and does nothing. This lets the simulation
        draw ground-truth values for the rocket bounding box on the display.
        '''
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.joystick_axes[event.axis] = event.value
            if event.type == pygame.JOYBUTTONDOWN:
                self._handle_button_press(event.button) 

        key = cv.waitKey(1)
        if self._auto_track_time is not None and time > self._auto_track_time and not self.tracking and not self._has_auto_tracked:
            print(f'Warning: auto-starting tracking at {self._auto_track_time}')
            key = ord('t')
            self._has_auto_tracked = True
        self._handle_keypress(key)
        
        img = self.environment.get_camera_image()
        if self._image_size is None:
            self._image_size = img.shape[1], img.shape[0]
            print(f"Image size: {self._image_size}")

        if img is None:
            img = np.zeros((1080,1920//2,3), dtype=np.uint8)
            cv.putText(img, "No camera image", (img.shape[1]//2, img.shape[0]//2), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            self.tracker.update_tracking(
                img,
                self.environment.get_telemetry(),
                time,
                self.tracking
            )

        if self.has_joystick and not self.tracking:
            self._joystick_control()
        
        # important: this has side effects on `img` so it has to be after the tracker update
        img_debug_callback(img)
        self._update_display(img, time)

        return True
            

