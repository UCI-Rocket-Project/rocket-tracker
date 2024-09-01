import traceback
from simulation.flight_model import test_flight
from src.component_algos.rocket_filter import RocketFilter
import numpy as np
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def format_numpy(x):
    return ", ".join([f"{i:.2f}" for i in x])

def plot_ellipsoid(center, singular_values, right_singular_vectors, ax, color):
    rotation = right_singular_vectors
    s = singular_values
    radii = np.sqrt(s)

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    return ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=color, linewidth=0.1, alpha=1, shade=True)

if __name__ == "__main__":
    # pad_geodetic_pos = np.array([35.347104, -117.808953, 620]) # the flight sim doesn't always use the _exact_ same position every time >:(
    pad_geodetic_pos = np.array([test_flight.latitude(0), test_flight.longitude(0), test_flight.z(0)])
    cam_geodetic_location = np.array([35.353056, -117.811944, 620])
    writer_gt = SummaryWriter(f'runs/rocket_filter/true')
    writer_pred = SummaryWriter(f'runs/rocket_filter/pred')


    pad_enu_pos = pm.geodetic2enu(*pad_geodetic_pos, *cam_geodetic_location)
    azimuth = np.rad2deg(np.arctan2(pad_enu_pos[1], pad_enu_pos[0]))
    altitude = np.rad2deg(np.arctan2(pad_enu_pos[2], np.linalg.norm(pad_enu_pos[:2])))
    filter = RocketFilter(
        pad_geodetic_pos, 
        cam_geodetic_location, 
        (azimuth, altitude),
        writer=writer_pred
    )

    start_geodetic = test_flight.latitude(0), test_flight.longitude(0), test_flight.z(0)
    print(start_geodetic)
    start_ecef = pm.geodetic2ecef(*start_geodetic)

    start_time = 0
    end_time = 70
    samples = 140
    dt = (end_time - start_time) / samples
    telemetry_period = 1
    last_telem = start_time
    err = []

    plot_data = []

    for i,t in tqdm(enumerate(np.linspace(start_time, end_time, samples))):
        copy_filter = filter.copy()
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

        xyz_enu = pm.ecef2enu(*xyz_ecef, *start_geodetic)
        azimuth, altitude = filter.hx_bearing(np.array([*xyz_ecef, 0,0,0,0,0]))

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
        writer_gt.add_scalar("bearing/altitude",altitude, i)

        try:
            filter.predict_update_bearing(t, np.array([azimuth, altitude]))


            if t - last_telem > telemetry_period:
                last_telem = t
                filter.predict_update_telem(
                    t,
                    np.array([
                        *(xyz_ecef + pos_noise),
                        test_flight.z(t)+altimeter_noise,
                    ])
                )
        except np.linalg.LinAlgError:
            traceback.print_exc()
            break
            

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
        writer_pred.add_scalar("thrust magnitude", filter.x[6], i)
        writer_pred.add_scalar("jerk", filter.x[7], i)
        pred_measurement = filter.hx_bearing(filter.x)
        writer_pred.add_scalar("bearing/azimuth", pred_measurement[0], i)
        writer_pred.add_scalar("bearing/altitude", pred_measurement[1], i)

        err.append(np.linalg.norm(filter.x[:3] - xyz_ecef))

        def make_plot_data():
            # make 3d plot of rocket path

            actual_path = []
            projected_path = []
            actual_vel  =[]
            projected_vel = []
            projected_accel = []
            projected_pos_covariance = []
            N_PREDICTIONS = 50
            dt = (end_time - t) / N_PREDICTIONS
            for future_t in np.linspace(t, end_time, N_PREDICTIONS):
                xyz_geodetic = test_flight.latitude(future_t), test_flight.longitude(future_t), test_flight.z(future_t)
                xyz_ecef = pm.geodetic2ecef(*xyz_geodetic)
                xyz_enu = pm.ecef2enu(*xyz_ecef, *start_geodetic)
                v_enu = test_flight.vx(future_t), test_flight.vy(future_t), test_flight.vz(future_t)
                actual_path.append(np.array(xyz_enu))
                actual_vel.append(np.array(v_enu))
                
                pred_ecef = copy_filter.x[:3]
                pred_enu = pm.ecef2enu(*pred_ecef, *start_geodetic)

                pred_vel_ecef = copy_filter.x[3:6]
                pred_vel_enu = pm.ecef2enu(*(pred_vel_ecef+start_ecef), *start_geodetic)
                projected_path.append(np.array(pred_enu))
                projected_vel.append(np.array(pred_vel_enu))

                projected_accel.append(copy_filter.x[6])
                projected_pos_covariance.append(copy_filter.P[:3,:3].copy())

                try:
                    copy_filter.predict(future_t)
                    # copy_filter.x = copy_filter.fx(copy_filter.x, dt)
                except np.linalg.LinAlgError:
                    print("Singular matrix")
                    break
            
            plot_data.append((t, actual_path, projected_path, actual_vel, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance))

        make_plot_data()

    t, actual_path, projected_path, actual_vel, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance = plot_data[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Rocket flight path vs UKF predictions')
    actual_position_marker = ax.scatter(*actual_path[0], color='C0', s=30)
    actual_path_line = ax.plot(*zip(*actual_path), label="actual path")
    projected_position_marker = ax.scatter(*projected_path[0], color='C1', s=20)
    projected_path_line = ax.plot(*zip(*projected_path), label="projected path", c='C1')
    projected_path_pts = ax.scatter(*zip(*projected_path), c='C1')

    covariance_ellipsoids = [
        plot_ellipsoid(pos, *np.linalg.svd(cov)[1:], ax, 'C1')
        for pos, cov in zip(projected_path[::5], projected_pos_covariance[::5])
    ]

    # arrows for velocity
    actual_path_quivers =  [
        ax.quiver(*pos, *(3*vel), color='C0') for pos, vel in zip(actual_path[::10], actual_vel[::10])
    ]

    projected_path_quivers = [
        ax.quiver(*pos, *(vel), color='C1') for pos, vel in zip(projected_path[::10], projected_vel[::10])
    ]

    cam_pos_enu = pm.geodetic2enu(*cam_geodetic_location, *start_geodetic)
    direction_vector =  actual_path[0] - cam_pos_enu
    bearing_line = ax.plot(*np.array([cam_pos_enu,cam_pos_enu+direction_vector*5]).T, color='C2', label='bearing from cam to rocket')

    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, f'Simulation timestep (* {end_time/samples:.2f}s)', 0, samples, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        t_index = int(slider.val)
        t, actual_path, projected_path, actual_vel, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance = plot_data[t_index]

        projected_path_line[0].set_data_3d(*zip(*projected_path))
        projected_path_pts._offsets3d = np.array(projected_path).T
        for i, (pos, vel) in enumerate(zip(projected_path[::10], projected_vel[::10])):
            projected_path_quivers[i].remove()
            projected_path_quivers[i] = ax.quiver(*pos, *(3*vel), color='C1')
        actual_position_marker._offsets3d = actual_path[0][:,None]
        direction_vector =  actual_path[0] - cam_pos_enu
        bearing_line[0].set_data_3d(*np.array([cam_pos_enu,cam_pos_enu+direction_vector*5]).T)
        projected_position_marker._offsets3d = projected_path[0][:,None]

        for i, (pos, cov) in enumerate(zip(projected_path[::5], projected_pos_covariance[::5])):
            covariance_ellipsoids[i].remove()
            covariance_ellipsoids[i] = plot_ellipsoid(pos, *np.linalg.svd(cov)[1:], ax, 'C1')

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(0, 4000)
    ax.legend(loc='upper left')
    plt.show()
    print(f"MSE: {np.mean(err)}, max: {np.max(err)}")
    print("Done")