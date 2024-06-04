from .joystick_commander import JoystickCommander
from .irl_environment import IRLEnvironment
from time import time

if __name__ == '__main__':
    env = IRLEnvironment()
    commander = JoystickCommander(env)
    start_time = time()
    while 1:
        commander.loop_callback(time()-start_time)