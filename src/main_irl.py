from .joystick_commander import JoystickCommander
#from .irl_environment import IRLEnvironment
# from .zr10_environment import ZR10Environment
from .usb_cam_environment import USBCamEnvironment
from torch.utils.tensorboard import SummaryWriter
from time import time

if __name__ == '__main__':
    # env = IRLEnvironment()
    # env = ZR10Environment()
    env = USBCamEnvironment()
    logger = SummaryWriter('runs/irl')
    commander = JoystickCommander(env, logger, vision_only=True)
    start_time = time()
    try:
        while 1:
            commander.loop_callback(time()-start_time)
    finally:
        env.move_telescope(0,0)
