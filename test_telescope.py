from telescope import Telescope
from time import sleep

telescope = Telescope(sim=False)

telescope.slewAzimuthRate(8.0)

sleep(1)

telescope.slewAzimuthRate(-8.0)

sleep(1)

telescope.slewAzimuthRate(0.0)