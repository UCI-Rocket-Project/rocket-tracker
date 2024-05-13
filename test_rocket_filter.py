from simulation.flight_model import test_flight
from rocket_filter import RocketFilter
import numpy as np
from matplotlib import pyplot as plt
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

if __name__ == "__main__":
    pad_geodetic_pos = np.array([35.35, -117.81, 620])
    writer_gt = SummaryWriter(f'runs/rocket_filter/true')
    writer_pred = SummaryWriter(f'runs/rocket_filter/pred')
    rocket = RocketFilter(pad_geodetic_pos)

    start_geodetic = test_flight.latitude(0), test_flight.longitude(0), test_flight.z(0)
    start_ecef = pm.geodetic2ecef(*start_geodetic)

    true_x = []
    pred_x = []

    start_time = 0
    end_time = 13
    samples = 50
    dt = (end_time - start_time) / samples
    for i,t in tqdm(enumerate(np.linspace(start_time, end_time, samples))):
        xyz_geodetic = test_flight.latitude(t), test_flight.longitude(t), test_flight.z(t)
        xyz_ecef = pm.geodetic2ecef(*xyz_geodetic)

        v_enu = test_flight.vx(t), test_flight.vy(t), test_flight.vz(t)
        v_ecef = np.array(pm.enu2ecef(*v_enu, *xyz_geodetic)) - np.array(pm.enu2ecef(0,0,0,*xyz_geodetic))

        accel_enu = test_flight.ax(t), test_flight.ay(t), test_flight.az(t)
        orientation = R.from_quat([test_flight.e0(t), test_flight.e1(t), test_flight.e2(t), test_flight.e3(t)])
        accel_body = orientation.apply(accel_enu)
        

        rocket.predict_update(
            dt,
            np.array([
                *accel_body,
                test_flight.w1(t),
                test_flight.w2(t),
                test_flight.w3(t),
                *xyz_ecef,
                test_flight.z(t)
            ]),
            (writer_pred, t)
        )


        true_x.append(xyz_ecef)
        pred_x.append(rocket.x[:3])
        writer_gt.add_scalar("ecef_x (offset from start)", xyz_ecef[0] - start_ecef[0], i)
        writer_gt.add_scalar("ecef_y (offset from start)", xyz_ecef[1] - start_ecef[1], i)
        writer_gt.add_scalar("ecef_z (offset from start)", xyz_ecef[2] - start_ecef[2], i)
        writer_gt.add_scalar("vx", v_ecef[0], i)
        writer_gt.add_scalar("vy", v_ecef[1], i)
        writer_gt.add_scalar("vz", v_ecef[2], i)
        writer_gt.add_scalar("q0", test_flight.e0(t), i)
        writer_gt.add_scalar("q1", test_flight.e1(t), i)
        writer_gt.add_scalar("q2", test_flight.e2(t), i)
        writer_gt.add_scalar("q3", test_flight.e3(t), i)
        writer_gt.add_scalar("wx", test_flight.w1(t), i)
        writer_gt.add_scalar("wy", test_flight.w2(t), i)
        writer_gt.add_scalar("wz", test_flight.w3(t), i)



        writer_pred.add_scalar("ecef_x (offset from start)", rocket.x[0] - start_ecef[0], i)
        writer_pred.add_scalar("ecef_y (offset from start)", rocket.x[1] - start_ecef[1], i)
        writer_pred.add_scalar("ecef_z (offset from start)", rocket.x[2] - start_ecef[2], i)
        writer_pred.add_scalar("vx", rocket.x[3], i)
        writer_pred.add_scalar("vy", rocket.x[4], i)
        writer_pred.add_scalar("vz", rocket.x[5], i)
        writer_pred.add_scalar("q0", rocket.x[6], i)
        writer_pred.add_scalar("q1", rocket.x[7], i)
        writer_pred.add_scalar("q2", rocket.x[8], i)
        writer_pred.add_scalar("q3", rocket.x[9], i)
        writer_pred.add_scalar("wx", rocket.x[10], i)
        writer_pred.add_scalar("wy", rocket.x[11], i)
        writer_pred.add_scalar("wz", rocket.x[12], i)
        writer_pred.add_scalar("thrust", rocket.x[13], i)
