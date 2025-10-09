# Rocket Tracker

https://github.com/UCI-Rocket-Project/rocket-tracker/assets/48658337/dc37688a-6238-4536-a412-5151f483befc

## Description:
Project that tracks rocket launches with a gimballed telescope. Doing this project involved:
- Writing python bindings for the camera drivers for an obscure astronomy camera with Cython
- Disilling an object detection foundation model to a smaller network to detect rockets in images and exporting that model to run fast on CPU
- Designing a kalman filter to predict rocket motion and fuse GPS telemetry with bounding boxes from object detection
- Making a custom simulator to integration test rocket dynamics predictions and computer vision
- Tuning a PID controller to control the camera gimbal to track aggressively

## Notes

This repo includes code for the rocket tracker, including drivers for the tripod mount and camera, and simulation of both the hardware and algorithm. Unfortunately, the code won't run with the latest versions of all the python libraries because neuralmagic deprecated Sparsify and I don't have the time or motivation to update the repo.

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
![project screenshot](https://github.com/user-attachments/assets/c4d861d7-fc96-4b68-a188-2d937a7022a3)
