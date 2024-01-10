import pygame
from telescope import Telescope

t = Telescope(sim=False)

pygame.init()
pygame.joystick.init()
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
    
    print(axis)

    if LEFT_AXIS_X not in axis:
        t.slewAzimuthRate(0)
        continue

    if axis[LEFT_AXIS_X] > 0.5:
        print("RIGHT")
        t.slewAzimuthRate(8)
    elif axis[LEFT_AXIS_X] < 0.5:
        print("LEFT")
        t.slewAzimuthRate(-8)
    else:
        t.slewAzimuthRate(0)