from telescope import Telescope

telescope = Telescope()

telescope.slew_rate_azi_alt(-8,0)

speed = 0
while speed < 8:
    speed = telescope.read_azi_speed()

print("-"*30)

telescope.slew_rate_azi_alt(8,0)

speed = 0
while speed < 8:
    speed = telescope.read_azi_speed()

telescope.slew_rate_azi_alt(0,6)

speed = 0
while speed < 6:
    speed = telescope.read_alt_speed()

print("-"*30)

telescope.slew_rate_azi_alt(0,-6)

speed = 0
while speed < 6:
    speed = telescope.read_alt_speed()

telescope.slew_rate_azi_alt(0,0)