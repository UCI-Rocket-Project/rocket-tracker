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
    filter = RocketFilter(
        pad_geodetic_pos, 
        cam_geodetic_location, 
        (azimuth, elevation),
        writer=writer_pred
    )

    start_geodetic = test_flight.latitude(0), test_flight.longitude(0), test_flight.z(0)
    print(start_geodetic)
    start_ecef = pm.geodetic2ecef(*start_geodetic)

    start_time = 0
    end_time = 60
    samples = 120
    dt = (end_time - start_time) / samples
    telemetry_period = 1
    last_telem = start_time
    err = []
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
        
        pos_noise = np.random.normal(0, 100, 3)
        altimeter_noise = np.random.normal(0, 1)

        rocket_enu_pos = pm.ecef2enu(*xyz_ecef, *cam_geodetic_location)
        azimuth = np.arctan2(rocket_enu_pos[1], rocket_enu_pos[0])
        elevation = np.arctan2(rocket_enu_pos[2], np.linalg.norm(rocket_enu_pos[:2]))

        xyz_enu = pm.geodetic2enu(*xyz_geodetic, *start_geodetic)
        writer_gt.add_scalar("enu position/x", xyz_enu[0], i)
        writer_gt.add_scalar("enu position/y", xyz_enu[1], i)
        writer_gt.add_scalar("enu position/z", xyz_enu[2], i)
        writer_gt.add_scalar("enu velocity/x", v_enu[0], i)
        writer_gt.add_scalar("enu velocity/y", v_enu[1], i)
        writer_gt.add_scalar("enu velocity/z", v_enu[2], i)
        writer_gt.add_scalar("enu acceleration/x", accel_enu[0], i)
        writer_gt.add_scalar("enu acceleration/y", accel_enu[1], i)
        writer_gt.add_scalar("enu acceleration/z", accel_enu[2], i)
        writer_gt.add_scalar("bearing/azimuth", azimuth, i)
        writer_gt.add_scalar("bearing/elevation", elevation, i)


        pred_ecef = filter.x[:3]
        pred_enu = pm.ecef2enu(*pred_ecef, *start_geodetic)

        pred_vel_ecef = filter.x[3:6]
        pred_vel_enu = pm.ecef2enu(*(pred_vel_ecef+start_ecef), *start_geodetic)

        writer_pred.add_scalar("enu position/x", pred_enu[0], i)
        writer_pred.add_scalar("enu position/y", pred_enu[1], i)
        writer_pred.add_scalar("enu position/z", pred_enu[2], i)
        writer_pred.add_scalar("enu velocity/x", pred_vel_enu[0], i)
        writer_pred.add_scalar("enu velocity/y", pred_vel_enu[1], i)
        writer_pred.add_scalar("enu velocity/z", pred_vel_enu[2], i)
        grav_vec = -9.81 * filter.x[:3] / np.linalg.norm(filter.x[:3])
        unit_v = filter.x[3:6] / np.linalg.norm(filter.x[3:6]) if np.linalg.norm(filter.x[3:6]) > 1 else -grav_vec / 9.81
        accel_ecef = unit_v * filter.x[6] + grav_vec
        accel_enu = pm.ecef2enu(*(accel_ecef+start_ecef), *pad_geodetic_pos)
        writer_pred.add_scalar("enu acceleration/x", accel_enu[0], i)
        writer_pred.add_scalar("enu acceleration/y", accel_enu[1], i)
        writer_pred.add_scalar("enu acceleration/z", accel_enu[2], i)
        writer_pred.add_scalar("jerk", filter.x[7], i)
        pred_measurement = filter.hx_bearing(filter.x)
        writer_pred.add_scalar("bearing/azimuth", pred_measurement[0], i)
        writer_pred.add_scalar("bearing/elevation", pred_measurement[1], i)


        filter.predict_update_bearing(t, np.array([azimuth, elevation]))


        if t - last_telem > telemetry_period:
            last_telem = t
            filter.predict_update_telem(
                t,
                np.array([
                    *(xyz_ecef + pos_noise),
                    test_flight.z(t)+altimeter_noise,
                ])
            )

        err.append(np.linalg.norm(filter.x[:3] - xyz_ecef))
    print(f"MSE: {np.mean(err)}")