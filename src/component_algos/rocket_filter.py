# import numpy as np
from functools import partial
import autograd.numpy as np
from autograd import jacobian
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from filterpy.kalman import ExtendedKalmanFilter
from itertools import product
from src.component_algos.depth_of_field import MM_PER_PIXEL

#https://kodlab.seas.upenn.edu/uploads/Arun/ekfpaper.pdf

def _copy_helper(obj, new_obj): # assign all float and ndarray attributes from obj to new_obj (side effect)
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        try:
            if isinstance(attr, float):
                setattr(new_obj, attr_name, attr)
            elif isinstance(attr, np.ndarray):
                setattr(new_obj, attr_name, attr.copy())
        except AttributeError: # some are @property functions that can't be set
            pass

class RocketFilter:
    STATE_DIM = 7
    ROCKET_HEIGHT = 6.5
    def __init__(self, 
                pad_geodetic_location: tuple[float,float,float], 
                cam_geodetic_location: tuple[float,float,float],
                initial_cam_orientation: tuple[float,float],
                focal_len_px: float,
                drag_coefficient: float = 5e-4,
                launch_time = None,
                writer: SummaryWriter = None):
        '''
        `pad_geodetic_location` is a tuple of (latitude, longitude, altitude) of the launchpad 
        It is used to initialize the filter state.
        if `launch_time` is not provided, you'll need to call `set_launch_time` before calling `predict_update`
        `cam_geodetic_location` is a tuple of (latitude, longitude, altitude) of the camera.
        This assumes the camera is exactly level so that azimuth doesn't change the vertical component of the bearing vector.
        '''
        # x dimension is [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz,thrust,dthrust]
        # xyz are in ECEF, velocities are in meters/sec
        # angular velocities are in body frame, in rad/sec
        # thrust is the acceleration due to the engine, in m/s^2
        # z dimension is [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, gps_x, gps_y, gps_z, altimeter]
        # GPS is in ECEF, altimeter is in meters above sea level

        self.pad_geodetic_location = pad_geodetic_location
        self.cam_geodetic_location = cam_geodetic_location
        self.drag_coefficient = drag_coefficient
        self.initial_cam_orientation = initial_cam_orientation
        self.writer = writer
        self.focal_len_px = focal_len_px

        self._launch_time = launch_time
        self._last_update_time = launch_time

        self._x_dim = RocketFilter.STATE_DIM
        self._z_dim = 6
        self.x = np.empty(self._x_dim) # state vector
        self.x[0:3] = pm.geodetic2ecef(*pad_geodetic_location)
        self.original_direction = self.x[:3] / np.linalg.norm(self.x[:3])
        self.x[3:6] = np.zeros(3) # velocity
        self.x[6] = 10 # linear acceleration

        position_std =  1e-2
        vel_std = 1e-2
        state_std = np.array([position_std, position_std, position_std, vel_std, vel_std, vel_std, 2])
        # assume we know the initial position to within 0.1m, velocity to within 0.01m/s, but acceleration
        # and jerk are less certain
        self.P = np.diag(np.square(state_std)) # state covariance matrix


        # assume position and velocity have little process noise, but acceleration and jerk have more
        pos_process_std = 1
        vel_process_std = 1
        process_std = np.array([pos_process_std, pos_process_std, pos_process_std, vel_process_std, vel_process_std, vel_process_std, 1])
        self.Q = np.diag(np.square(process_std)) # process noise covariance matrix

        # assume GPS is accurate to within 100m, altimeter is accurate to within 1m
        gps_pos_std_meters = 1
        gps_vel_std_meters = 1e-4
        telem_measurement_std = 1*np.array([
            gps_pos_std_meters,gps_pos_std_meters,gps_pos_std_meters,
            gps_vel_std_meters,gps_vel_std_meters,gps_vel_std_meters
        ])
        self.R_telem = np.diag(np.square(telem_measurement_std)) # measurement noise covariance matrix

        bearing_measurement_std = np.array([1e-2, 1e-2, 1])
        self.R_bearing = np.diag(np.square(bearing_measurement_std)) # measurement noise covariance matrix




        self.telem_ekf = ExtendedKalmanFilter(
            dim_x=self._x_dim,
            dim_z=self._z_dim,
        )
        self.telem_ekf.x = self.x
        self.telem_ekf.P = self.P
        self.telem_ekf.Q = self.Q
        self.telem_ekf.R = self.R_telem

        self.bearing_ekf = ExtendedKalmanFilter(
            dim_x=self._x_dim,
            dim_z=3,
        )
        self.bearing_ekf.x = self.x
        self.bearing_ekf.P = self.P
        self.bearing_ekf.Q = self.Q
        self.bearing_ekf.R = self.R_bearing
        
        self.flight_time = 0


    def hx_bearing(self, x: np.ndarray):
        '''
        Measurement function for camera bearing measurements. In future versions,
        we could also add measurements of the rocket's apparent size and orientation.
        '''

        rocket_initial_enu_pos = np.array(pm.geodetic2enu(*self.pad_geodetic_location, *self.cam_geodetic_location))
        rocket_pos_enu = np.array(pm.ecef2enu(*x[:3], *self.cam_geodetic_location))
        initial_pos_xy = rocket_initial_enu_pos[:2]
        # autograd raises an exception when using np.linalg.norm so we have to do this manually
        def norm(x): 
            return np.sqrt(np.sum(np.square(x)))
        initial_pos_xy /= norm(initial_pos_xy)
        rocket_pos_xy = rocket_pos_enu[:2]
        rocket_pos_xy /= norm(rocket_pos_xy)
        
        # im a dumbass so I didn't figure out this math myself
        # https://stackoverflow.com/a/16544330
        dot = np.dot(initial_pos_xy, rocket_pos_xy)
        # autograd also crashes when trying to use np.cross
        # equivalent to `det=np.cross(initial_pos_xy, rocket_pos_xy)`
        det = initial_pos_xy[0] * rocket_pos_xy[1] - initial_pos_xy[1] * rocket_pos_xy[0]
        theta = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        azimuth_bearing = self.initial_cam_orientation[0] + np.rad2deg(theta)
        elevation_bearing = np.rad2deg(np.arctan2(rocket_pos_enu[2], norm(rocket_pos_enu[:2])))

        # TODO: rewrite this so it uses better math than similar triangles which assumes the rocket is in the center of the frame
        dist_to_cam = norm(rocket_pos_enu)
        apparent_size = self.focal_len_px * RocketFilter.ROCKET_HEIGHT / dist_to_cam

        return np.array([
            azimuth_bearing,
            elevation_bearing,
            apparent_size
        ])

    def hx_telem(self, x: np.ndarray):
        '''
        Measurement function for telemetry measurements
        '''

        return np.array([
            x[0], x[1], x[2], # ECEF position
            x[3], x[4], x[5], # ECEF velocity
        ])
    
    def fx(self, x: np.ndarray, dt: float):
        '''
        State transition function
        '''
        grav_vec = -9.81 * x[:3] / np.linalg.norm(x[:3])
        vel_magnitude = np.linalg.norm(x[3:6])
        # thrust direction is unit vector in velocity direction, or straight up if velocity is low (for initial lift off the launchpad)
        thrust_direction = x[3:6] / vel_magnitude if vel_magnitude > 10 else -grav_vec / 9.81
        drag = -self.drag_coefficient * np.sum(np.square(x[3:6])) * thrust_direction
        # if for drag we do vel_magnitude**2, it blows up the gradient when velocity is zero
        # because of the square root in the norm.
        accel = thrust_direction * np.abs(x[6]) + grav_vec + drag
        return np.array([
            *(x[0:3] + x[3:6] * dt + 0.5 * accel * dt**2),
            *(x[3:6] + accel * dt),
            x[6] if self.flight_time < 20 else 0
        ])

    def set_launch_time(self, time: float):
        if self._launch_time is not None:
            raise RuntimeError('Trying to set launch time when it has already been set')
        self._launch_time = time
        self._last_update_time = time


    def _log_state(self, time: float):
        if self.writer is None:
            raise RuntimeError('Trying to log state without a SummaryWriter passed to the filter')
        predicted_az, predicted_alt, predicted_size = self.hx_bearing(self.bearing_ekf.x)
        self.writer.add_scalar('ekf/azi_predicted', predicted_az, time*100)
        self.writer.add_scalar('ekf/alt_predicted', predicted_alt, time*100)
        self.writer.add_scalar('ekf/size_predicted', predicted_size, time*100)
        for i, x in enumerate(self.x):
            self.writer.add_scalar(f'ekf/x_{i}', x, time*100)
        for i,j in product(range(self.P.shape[0]), range(self.P.shape[1])):
            self.writer.add_scalar(f'ekf/P_{i}_{j}', self.P[i,j], time*100)
        self.writer.add_scalar('ekf/bearing_log_likelihood', self.bearing_ekf.log_likelihood, time*100)
        self.writer.add_scalar('ekf/telem_log_likelihood', self.telem_ekf.log_likelihood, time*100)

    def predict_update_bearing(self, time: float, z: np.ndarray):
        '''
        Predict and update the filter with a bearing measurement
        '''
        if self._launch_time is None:
            raise RuntimeError('Trying to predict and update filter without setting launch time')

        time_since_first_update = time - self._launch_time
        dt = time_since_first_update - self._last_update_time
        self.flight_time = time_since_first_update
        self.bearing_ekf.F = jacobian(partial(self.fx, dt=dt))(self.x)
        self.bearing_ekf.predict()
        self._last_update_time = time_since_first_update
        # not sure if the jacobian should be calculated with x before or after the prediction
        self.bearing_ekf.update(z, HJacobian=jacobian(self.hx_bearing), Hx=self.hx_bearing)
        self.x = self.bearing_ekf.x
        self.P = self.bearing_ekf.P
        self.telem_ekf.x = self.bearing_ekf.x
        self.telem_ekf.P = self.bearing_ekf.P
        if self.writer is not None:
            self.writer.add_scalar('ekf/azi_measured', z[0], time*100)
            self.writer.add_scalar('ekf/alt_measured', z[1], time*100) 
            self._log_state(time)
    
    def predict_update_telem(self, time: float, z: np.ndarray):
        '''
        debug_logging is a tuple of (SummaryWriter, int) where the int is the current iteration number
        '''

        if self._launch_time is None:
            raise RuntimeError('Trying to predict and update filter without setting launch time')
        time_since_first_update = time - self._launch_time
        dt = time_since_first_update - self._last_update_time
        self.flight_time = time_since_first_update
        self._last_update_time = time_since_first_update
        self.telem_ekf.F = jacobian(partial(self.fx, dt=dt))(self.x)
        self.telem_ekf.predict()
        self.telem_ekf.update(z, HJacobian=jacobian(self.hx_telem), Hx=self.hx_telem)
        self.x = self.telem_ekf.x
        self.P = self.telem_ekf.P
        self.bearing_ekf.x = self.telem_ekf.x
        self.bearing_ekf.P = self.telem_ekf.P
        if self.writer is not None:
            self._log_state(time)

    def predict(self, time_since_first_update: float):
        '''
        Predict the filter state
        '''
        dt = time_since_first_update - self._last_update_time
        self.flight_time = time_since_first_update
        self._last_update_time = time_since_first_update
        self.telem_ekf.x = self.x
        self.telem_ekf.F = jacobian(partial(self.fx, dt=dt))(self.x)
        self.telem_ekf.predict(dt)
        self.x = self.telem_ekf.x
        self.P = self.telem_ekf.P
        self.bearing_ekf.x = self.telem_ekf.x
        self.bearing_ekf.P = self.telem_ekf.P
                
    def _quat_mult(self, q1: np.ndarray, q2: np.ndarray):
        '''
        Multiply two quaternions
        '''
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _to_quat(self, w_q: np.ndarray):
        '''
        Convert rotation vectors to quaternions
        v is (3, n)
        returns (4,n) array of quaternions
        '''
        alpha_w = np.linalg.norm(w_q, axis=0)
        div_vals = np.where(alpha_w == 0, 1, alpha_w) # avoid division by zero
        # if alpha_w is zero, then the e_w values are all multiplied by sin(0) = 0 so their values dont' matter.
        e_w = w_q / div_vals
        q_w = np.array([
            np.cos(alpha_w/2),
            e_w[0] * np.sin(alpha_w/2),
            e_w[1] * np.sin(alpha_w/2),
            e_w[2] * np.sin(alpha_w/2)
        ])

        return q_w

    def _avg_quaternions(self, Q: np.ndarray):
        '''
        Q is a (n,4) array of quaternions

        This differs from the iterative method in the paper by Kraft. Instead,
        we use the eigenvector method from this paper:
        http://www.acsu.buffalo.edu/%7Ejohnc/ave_quat07.pdf
        '''

        assert Q.shape[1] == 4, "Q must have 4 columns"

        _, eigenvectors = np.linalg.eig(Q.T @ Q)

        return eigenvectors[:,0] / np.linalg.norm(eigenvectors[:,0])
    


    def copy(self):
        '''
        Create a copy of the filter. Doesn't copy the writer, and updating the copy doesn't update the original
        '''
        new_filter = RocketFilter(
            self.pad_geodetic_location,
            self.cam_geodetic_location,
            self.initial_cam_orientation,
            self.drag_coefficient,
            None
        )
        _copy_helper(self, new_filter)

        def _copy_ekf(ekf: ExtendedKalmanFilter):
            new_ekf =  ExtendedKalmanFilter(
                dim_x=self._x_dim,
                dim_z=self._z_dim,
            )
            _copy_helper(ekf, new_ekf)
            # go through every variable, if its float copy directly, if ndarray copy with copy(), otherwise ignore
            return new_ekf

        new_filter.telem_ekf = _copy_ekf(self.telem_ekf)
        new_filter.bearing_ekf = _copy_ekf(self.bearing_ekf)

        return new_filter