# Rocket Tracker

Repo for the code for the rocket tracker, including drivers for the tripod mount and camera, and simulation of both the hardware and algorithm.

The hardware simulator is here: https://github.com/ASCOMInitiative/ASCOM.Alpaca.Simulators/releases

## Installing camera software:
1. Download ASIStudio (https://www.zwoastro.com/downloads/linux)
2. make the `.run` file executable, and run it
3. (optional) run ASIImg to get a preview. Some settings I found that work decently are:
     - 1920x1080 resolution
     - 0.02 exposure
     - L(0) gain
     - Bin1