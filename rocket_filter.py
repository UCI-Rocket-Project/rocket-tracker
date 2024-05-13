import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

#https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf

class RocketFilter:
    def __init__(self, pad_geodetic_location: tuple[float,float,float]):
        '''
        `pad_geodetic_location` is a tuple of (latitude, longitude, altitude) of the launchpad 
        It is used to initialize the filter state.
        '''
        # x dimension is [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz,thrust,dthrust]
        # xyz are in ECEF, velocities are in meters/sec
        # angular velocities are in body frame, in rad/sec
        # thrust is the acceleration due to the engine, in m/s^2
        # z dimension is [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, gps_x, gps_y, gps_z, altimeter]
        # GPS is in ECEF, altimeter is in meters above sea level

        x_dim = 8
        z_dim = 4
        self.x = np.empty(x_dim) # state vector
        self.x[0:3] = pm.geodetic2ecef(*pad_geodetic_location)
        self.original_direction = self.x[:3] / np.linalg.norm(self.x[:3])
        self.x[3:6] = np.zeros(3) # velocity
        self.x[6] = 9.81 # linear acceleration
        self.x[7] = 1 # linear jerk

        # covariance matrices have 1 less dimension because the quaternion orientation
        # is 4 variables, but only 3 degrees of freedom
        self.P = np.eye(x_dim) # state covariance matrix
        self.Q = 0.1* np.eye(x_dim) # process noise covariance matrix
        self.R = 1e-3 * np.eye(z_dim) # measurement noise covariance matrix

        self.ukf = UnscentedKalmanFilter(
            dim_x=x_dim,
            dim_z=z_dim,
            dt=1,
            hx=self.hx,
            fx=self.fx,
            points=MerweScaledSigmaPoints(n=x_dim, alpha=1e-3, beta=2, kappa=0)
        )
        self.ukf.x = self.x
        self.ukf.P = self.P
        self.ukf.Q = self.Q
        self.ukf.R = self.R


    def hx(self, x: np.ndarray):
        '''
        Measurement function
        '''

        geodetic = pm.ecef2geodetic(x[0], x[1], x[2])

        return np.array([
            x[0], x[1], x[2], # ECEF position
            geodetic[2] # altitude
        ])
    
    def fx(self, x: np.ndarray, dt: float):
        '''
        State transition function
        '''
        grav_vec = -9.81 * x[:3] / np.linalg.norm(x[:3])
        vel_unit = x[3:6] / np.linalg.norm(x[3:6]) if np.linalg.norm(x[3:6]) > 1 else -grav_vec / 9.81
        accel = vel_unit * x[6] + grav_vec
        jerk = vel_unit * x[7]
        x[0:3] += x[3:6] * dt + 0.5 * accel * dt**2 + 1/6 * jerk * dt**3
        x[3:6] += accel * dt + 0.5 * jerk * dt**2
        x[6] += x[7] * dt
        return x
    
    def predict_update(self, dt: float, z: np.ndarray, debug_logging: tuple[SummaryWriter, int] = None):
        '''
        debug_logging is a tuple of (SummaryWriter, int) where the int is the current iteration number
        '''
        self.ukf.predict(dt)
        self.ukf.update(z)
        self.x = self.ukf.x
        self.P = self.ukf.P
                
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