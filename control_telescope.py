import pygame
from telescope import Telescope
import numpy as np

t = Telescope()

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

# main loop
while True:
    # retrieve any events ...
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            axis[event.axis] = event.value
    
    slew_x = 0
    slew_y = 0
    
    if LEFT_AXIS_X in axis and abs(axis[LEFT_AXIS_X])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_X]) # use tanh to clamp to [-1,1]
        slew_x = clamped*8
    
    if LEFT_AXIS_Y in axis and abs(axis[LEFT_AXIS_Y])>0.1:
        clamped = np.tanh(axis[LEFT_AXIS_Y]) # use tanh to clamp to [-1,1]
        slew_y = clamped*6

    t.slew_rate_azi_alt(slew_x, slew_y)