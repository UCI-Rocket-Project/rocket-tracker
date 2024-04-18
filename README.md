# Rocket Tracker

Repo for the code for the rocket tracker, including drivers for the tripod mount and camera, and simulation of both the hardware and algorithm.

The hardware simulator is here: https://github.com/ASCOMInitiative/ASCOM.Alpaca.Simulators/releases

Panda3D install: https://docs.panda3d.org/1.10/python/introduction/

To run the simulation, do `LD_LIBRARY_PATH='/home/eric/code/py-zwo-eaf/src/EAF_linux_mac_SDK_V1.6/lib/x64' py -m simulation.sim`.

The exact library path will change based on where the autofocuser library is installed to.

To run the tracker IRL, do `python3 control_telescope.py`

## Installing camera software:
https://pypi.org/project/camera-zwo-asi/