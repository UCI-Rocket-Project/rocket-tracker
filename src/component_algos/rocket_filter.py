import numpy as np
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

#https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf

class RocketFilter:
    def __init__(self, 
                pad_geodetic_location: tuple[float,float,float], 
                cam_geodetic_location: tuple[float,float,float],
                initial_cam_orientation: tuple[float,float],
                drag_coefficient: float = 5e-4,
                writer: SummaryWriter = None):
        '''
        `pad_geodetic_location` is a tuple of (latitude, longitude, altitude) of the launchpad 
        It is used to initialize the filter state.
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

        self.last_update_time = 0

        x_dim = 8
        z_dim = 4
        self.x = np.empty(x_dim) # state vector
        self.x[0:3] = pm.geodetic2ecef(*pad_geodetic_location)
        self.original_direction = self.x[:3] / np.linalg.norm(self.x[:3])
        self.x[3:6] = np.zeros(3) # velocity
        self.x[6] = 10 # linear acceleration
        self.x[7] = 5 # linear jerk

        state_variances = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 5, 2])
        # assume we know the initial position to within 0.1m, velocity to within 0.01m/s, but acceleration
        # and jerk are less certain
        self.P = np.diag(state_variances) # state covariance matrix


        # assume position and velocity have little process noise, but acceleration and jerk have more
        process_variances = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.1, 10])
        self.Q = np.diag(process_variances) # process noise covariance matrix

        # assume GPS is accurate to within 100m, altimeter is accurate to within 1m
        telem_measurement_variances = np.array([100,100,100,1])
        self.R_telem = np.diag(telem_measurement_variances) # measurement noise covariance matrix

        bearing_measurement_variances = np.array([1e-6, 1e-6])
        self.R_bearing = np.diag(bearing_measurement_variances) # measurement noise covariance matrix


        self.telem_ukf = UnscentedKalmanFilter(
            dim_x=x_dim,
            dim_z=z_dim,
            dt=1,
            hx=self.hx_telem,
            fx=self.fx,
            points=MerweScaledSigmaPoints(n=x_dim, alpha=1e-3, beta=2, kappa=0)
        )
        self.telem_ukf.x = self.x
        self.telem_ukf.P = self.P
        self.telem_ukf.Q = self.Q
        self.telem_ukf.R = self.R_telem

        self.bearing_ukf = UnscentedKalmanFilter(
            dim_x=x_dim,
            dim_z=2,
            dt=1,
            hx=self.hx_bearing,
            fx=self.fx,
            points=MerweScaledSigmaPoints(n=x_dim, alpha=1e-3, beta=2, kappa=0)
        )
        self.bearing_ukf.x = self.x
        self.bearing_ukf.P = self.P
        self.bearing_ukf.Q = self.Q
        self.bearing_ukf.R = self.R_bearing
        
        self.flight_time = 0


    def hx_bearing(self, x: np.ndarray):
        '''
        Measurement function for camera bearing measurements. In future versions,
        we could also add measurements of the rocket's apparent size and orientation.
        '''

        rocket_pos_enu = pm.ecef2enu(*x[:3], *self.cam_geodetic_location)
        azimuth_bearing = np.arctan2(rocket_pos_enu[1], rocket_pos_enu[0]) + self.initial_cam_orientation[0]
        elevation_bearing = np.arctan2(rocket_pos_enu[2], np.linalg.norm(rocket_pos_enu[:2])) + self.initial_cam_orientation[1]
        

        return np.array([
            azimuth_bearing,
            elevation_bearing
        ])

    def hx_telem(self, x: np.ndarray):
        '''
        Measurement function for telemetry measurements
        '''

        geodetic = pm.ecef2geodetic(x[0], x[1], x[2])

        return np.array([
            x[0], x[1], x[2], # ECEF position
            geodetic[2], # altitude,
        ])
    
    def fx(self, x: np.ndarray, dt: float):
        '''
        State transition function
        '''
        grav_vec = -9.81 * x[:3] / np.linalg.norm(x[:3])
        vel_magnitude = np.linalg.norm(x[3:6])
        thrust_direction = x[3:6] / vel_magnitude if vel_magnitude > 10 else -grav_vec / 9.81
        jerk = thrust_direction * x[7]
        drag = -self.drag_coefficient * vel_magnitude**2 * thrust_direction
        if self.flight_time > 1e4:
            accel = grav_vec+drag
        else:
            accel = thrust_direction * x[6] + grav_vec + drag
            jerk = 0
        x[0:3] += x[3:6] * dt + 0.5 * accel * dt**2 + 1/6 * jerk * dt**3
        x[3:6] += accel * dt + 0.5 * jerk * dt**2
        x[6] += x[7] * dt
        return x

    def predict_update_bearing(self, time_since_first_update: float, z: np.ndarray):
        '''
        Predict and update the filter with a bearing measurement
        '''
        dt = time_since_first_update - self.last_update_time
        self.flight_time = time_since_first_update
        self.bearing_ukf.predict(dt)
        self.last_update_time = time_since_first_update
        self.bearing_ukf.update(z)
        self.x = self.bearing_ukf.x
        self.P = self.bearing_ukf.P
        self.telem_ukf.x = self.bearing_ukf.x
        self.telem_ukf.P = self.bearing_ukf.P
    
    def predict_update_telem(self, time_since_first_update: float, z: np.ndarray):
        '''
        debug_logging is a tuple of (SummaryWriter, int) where the int is the current iteration number
        '''
        dt = time_since_first_update - self.last_update_time
        self.flight_time = time_since_first_update
        self.last_update_time = time_since_first_update
        self.telem_ukf.predict(dt)
        self.telem_ukf.update(z)
        self.x = self.telem_ukf.x
        self.P = self.telem_ukf.P
        self.bearing_ukf.x = self.telem_ukf.x
        self.bearing_ukf.P = self.telem_ukf.P

    def predict(self, time_since_first_update: float):
        '''
        Predict the filter state
        '''
        dt = time_since_first_update - self.last_update_time
        self.flight_time = time_since_first_update
        self.last_update_time = time_since_first_update
        self.telem_ukf.predict(dt)
        self.x = self.telem_ukf.x
        self.P = self.telem_ukf.P
        self.bearing_ukf.x = self.telem_ukf.x
        self.bearing_ukf.P = self.telem_ukf.P
                
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