# Rocket Tracker

https://github.com/UCI-Rocket-Project/rocket-tracker/assets/48658337/dc37688a-6238-4536-a412-5151f483befc



Repo for the code for the rocket tracker, including drivers for the tripod mount and camera, and simulation of both the hardware and algorithm.

The hardware simulator is here: https://github.com/ASCOMInitiative/ASCOM.Alpaca.Simulators/releases

Panda3D install: https://docs.panda3d.org/1.10/python/introduction/


Problem with ld library path solution:
`LD_LIBRARY_PATH='/home/eric/code/py-zwo-eaf/src/EAF_linux_mac_SDK_V1.6/lib/x64' py -m simulation.main`.

Commands:
py -m simulation.main
py -m src.component_algos.visualize
py -m src.component_algos.test_rocket_filter

The exact library path will change based on where the autofocuser library is installed to.

To run the tracker IRL, do `python3 control_telescope.py`

## Installing camera software:
https://pypi.org/project/camera-zwo-asi/
