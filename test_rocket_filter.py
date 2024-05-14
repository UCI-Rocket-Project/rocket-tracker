from simulation.flight_model import test_flight
from rocket_filter import RocketFilter
import numpy as np
from matplotlib import pyplot as plt
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

if __name__ == "__main__":
    pad_geodetic_pos = np.array([35.347104, -117.808953, 620])
    cam_geodetic_location = np.array([35.353056, -117.811944, 620])
    writer_gt = SummaryWriter(f'runs/rocket_filter/true')
    writer_pred = SummaryWriter(f'runs/rocket_filter/pred')


    pad_enu_pos = pm.geodetic2enu(*pad_geodetic_pos, *cam_geodetic_location)
    azimuth = np.arctan2(pad_enu_pos[1], pad_enu_pos[0])
    elevation = np.arctan2(pad_enu_pos[2], np.linalg.norm(pad_enu_pos[:2]))
    rocket = RocketFilter(pad_geodetic_pos, cam_geodetic_location, (azimuth, elevation))

    start_geodetic = test_flight.latitude(0), test_flight.longitude(0), test_flight.z(0)
    start_ecef = pm.geodetic2ecef(*start_geodetic)

    true_x = []
    pred_x = []

    start_time = 0
    end_time = 60
    samples = 30
    dt = (end_time - start_time) / samples
    for i,t in tqdm(enumerate(np.linspace(start_time, end_time, samples))):
        xyz_geodetic = test_flight.latitude(t), test_flight.longitude(t), test_flight.z(t)
        xyz_ecef = pm.geodetic2ecef(*xyz_geodetic)

        v_enu = test_flight.vx(t), test_flight.vy(t), test_flight.vz(t)
        v_ecef = np.array(pm.enu2ecef(*v_enu, *xyz_geodetic)) - np.array(pm.enu2ecef(0,0,0,*xyz_geodetic))

        accel_enu = np.array([test_flight.ax(t), test_flight.ay(t), test_flight.az(t)])
        accel_ecef = np.array(pm.enu2ecef(*accel_enu, *xyz_geodetic)) - np.array(pm.enu2ecef(0,0,0,*xyz_geodetic))
        orientation = R.from_quat([test_flight.e0(t), test_flight.e1(t), test_flight.e2(t), test_flight.e3(t)])
        # orientation.apply([0,0,1]) is the direction of the rocket in ENU
        gravity_ecef = np.array(pm.enu2ecef(0,0,-9.81,*xyz_geodetic)) - np.array(pm.enu2ecef(0,0,0,*xyz_geodetic))
        accel_body = orientation.apply(accel_enu + np.array([0,0,9.81]))
        compensated_accel = accel_body - gravity_ecef
        accel_ecef_unit = (compensated_accel) / np.linalg.norm(compensated_accel)
        v_ecef_unit = v_ecef / np.linalg.norm(v_ecef)
        
        pos_noise = np.random.normal(0, 100, 3)
        altimeter_noise = np.random.normal(0, 1)

        rocket_enu_pos = pm.ecef2enu(*rocket.x[:3], *pad_geodetic_pos)
        azimuth = np.arctan2(rocket_enu_pos[1], rocket_enu_pos[0])
        elevation = np.arctan2(rocket_enu_pos[2], np.linalg.norm(rocket_enu_pos[:2]))
        rocket.predict_update_bearing(t, np.array([azimuth, elevation]))

        # rocket.predict_update_telem(
        #     t,
        #     np.array([
        #         *(xyz_ecef + pos_noise),
        #         test_flight.z(t)+altimeter_noise,
        #     ])
        # )


        true_x.append(xyz_ecef)
        pred_x.append(rocket.x[:3])
        writer_gt.add_scalar("ecef offset/x", xyz_ecef[0] - start_ecef[0], i)
        writer_gt.add_scalar("ecef offset/y", xyz_ecef[1] - start_ecef[1], i)
        writer_gt.add_scalar("ecef offset/z", xyz_ecef[2] - start_ecef[2], i)
        writer_gt.add_scalar("velocity/x", v_ecef[0], i)
        writer_gt.add_scalar("velocity/y", v_ecef[1], i)
        writer_gt.add_scalar("velocity/z", v_ecef[2], i)
        writer_gt.add_scalar("acceleration/x", accel_ecef[0], i)
        writer_gt.add_scalar("acceleration/y", accel_ecef[1], i)
        writer_gt.add_scalar("acceleration/z", accel_ecef[2], i)



        writer_pred.add_scalar("ecef offset/x", rocket.x[0] - start_ecef[0], i)
        writer_pred.add_scalar("ecef offset/y", rocket.x[1] - start_ecef[1], i)
        writer_pred.add_scalar("ecef offset/z", rocket.x[2] - start_ecef[2], i)
        writer_pred.add_scalar("velocity/x", rocket.x[3], i)
        writer_pred.add_scalar("velocity/y", rocket.x[4], i)
        writer_pred.add_scalar("velocity/z", rocket.x[5], i)
        unit_v = rocket.x[3:6] / np.linalg.norm(rocket.x[3:6])
        grav_vec = -9.81 * rocket.x[:3] / np.linalg.norm(rocket.x[:3])
        accel = unit_v * rocket.x[6] + grav_vec
        writer_pred.add_scalar("acceleration/x", accel[0], i)
        writer_pred.add_scalar("acceleration/y", accel[1], i)
        writer_pred.add_scalar("acceleration/z", accel[2], i)
        writer_pred.add_scalar("jerk", rocket.x[7], i)