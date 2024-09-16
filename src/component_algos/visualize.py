from src.component_algos.read_tensorboard import logs_dir_to_dataframe
from src.component_algos.rocket_filter import RocketFilter
import matplotlib.pyplot as plt
import pymap3d as pm
import pandas as pd
import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import argparse

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

def visualize_df(df: pd.DataFrame):
    actual_path = []
    actual_vel = []
    
    plot_data = []
    first_filter_x = None


    cam_geodetic_location = np.array([35.353056, -117.811944, 620])
    row0 = df.iloc[0]
    rocket_start_ecef = np.array([row0['pred/ukf/x_0'], row0['pred/ukf/x_1'], row0['pred/ukf/x_2']])
    pad_geodetic_location = pm.ecef2geodetic(*rocket_start_ecef)

    cam_initial_bearing = (row0['pred/ukf/azi_measured'], row0['pred/ukf/alt_measured'])

    for time, row in df.iterrows():
        enu_x = row['true/enu position/x']
        enu_y = row['true/enu position/y']
        enu_z = row['true/enu position/z']

        v_enu_x = row['true/enu velocity/x']
        v_enu_y = row['true/enu velocity/y']
        v_enu_z = row['true/enu velocity/z']

        actual_path.append(np.array([enu_x, enu_y, enu_z]))
        actual_vel.append(np.array([v_enu_x, v_enu_y, v_enu_z]))

        filter_x = np.array([row[f'pred/ukf/x_{i}'] for i in range(RocketFilter.STATE_DIM)])
        filter_P = np.array([row[f'pred/ukf/P_{i}_{j}'] for i,j in product(range(RocketFilter.STATE_DIM), range(RocketFilter.STATE_DIM))]).reshape(RocketFilter.STATE_DIM, RocketFilter.STATE_DIM)

        filter = RocketFilter(
            pad_geodetic_location,
            cam_geodetic_location,
            cam_initial_bearing,
            launch_time=0,
        )

        filter.x = filter_x
        filter.P = filter_P

        projected_path = []
        projected_vel = []
        projected_accel = []
        projected_pos_covariance = []

        dt = 0.5

        for i in range(0, 50):
            pred_ecef = filter.x[:3]
            pred_enu = pm.ecef2enu(*pred_ecef, *pad_geodetic_location)

            pred_vel_ecef = filter.x[3:6]
            pred_vel_enu = pm.ecef2enu(*(pred_vel_ecef+rocket_start_ecef), *pad_geodetic_location)
            projected_path.append(np.array(pred_enu))
            projected_vel.append(np.array(pred_vel_enu))

            projected_accel.append(filter.x[6])
            projected_pos_covariance.append(filter.P[:3,:3].copy())

            try:
                filter.predict(i*dt)
            except np.linalg.LinAlgError:
                print("Singular matrix")
                break
        plot_data.append(
            [time, projected_path, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance]
        )

    t, projected_path, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance = plot_data[0]
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

    cam_pos_enu = pm.geodetic2enu(*cam_geodetic_location, *pad_geodetic_location)
    direction_vector =  actual_path[0] - cam_pos_enu
    bearing_line = ax.plot(*np.array([cam_pos_enu,cam_pos_enu+direction_vector*5]).T, color='C2', label='bearing from cam to rocket')

    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, f'Simulation timestep (index, no units)', 0, len(df), valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        t_index = int(slider.val)
        t, projected_path, projected_vel, pred_vel_enu, projected_accel, projected_pos_covariance = plot_data[t_index]
        gt_pos = actual_path[t_index]

        projected_path_line[0].set_data_3d(*zip(*projected_path))
        projected_path_pts._offsets3d = np.array(projected_path).T
        for i, (pos, vel) in enumerate(zip(projected_path[::10], projected_vel[::10])):
            projected_path_quivers[i].remove()
            projected_path_quivers[i] = ax.quiver(*pos, *(3*vel), color='C1')
        actual_position_marker._offsets3d = gt_pos[:,None]
        direction_vector =  gt_pos - cam_pos_enu
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize logs')
    parser.add_argument('logs_path', nargs='?', default='runs/test_rocket_filter', help='path to tensorboard logs (defualt is runs/test_rocket_filter)')

    args = parser.parse_args()

    df = logs_dir_to_dataframe(args.logs_path)
    df = df[df['pred/ukf/x_0'].notna()]
    visualize_df(df)