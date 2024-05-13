import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky
import pymap3d as pm
from torch.utils.tensorboard import SummaryWriter

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

        x_dim = 15
        z_dim = 10
        self.x = np.empty(x_dim) # state vector
        self.x[0:3] = pm.geodetic2ecef(*pad_geodetic_location)
        self.original_direction = self.x[:3] / np.linalg.norm(self.x[:3])
        self.x[3:6] = np.zeros(3) # velocity
        # initial_point_direction = self.x[:3] / np.linalg.norm(self.x[:3])
        # initial_rot = R.align_vectors(np.array([0,0,1])[None,:], initial_point_direction[None,:])[0]
        self.x[6:10] = R.identity().as_quat()
        self.x[10:13] = np.zeros(3) # angular velocity
        self.x[13] = 1 # thrust
        self.x[14] = 1 # dthrust

        # covariance matrices have 1 less dimension because the quaternion orientation
        # is 4 variables, but only 3 degrees of freedom
        self.P = np.eye(x_dim - 1) # state covariance matrix
        process_noise_covariances = 1e-3 * np.array([
            0.1, 0.1, 0.1, # position
            0.1, 0.1, 0.1, # velocity
            0.1, 0.1, 0.1, # orientation
            0.1, 0.1, 0.1, # angular velocity
            0.1, 0.1 # thrust and dthrust
        ])
        self.Q = np.diag(process_noise_covariances) # process noise covariance matrix

        self.R = 1e-3 * np.eye(z_dim) # measurement noise covariance matrix


    def hx(self, x: np.ndarray):
        '''
        Measurement function
        '''

        acc_vector = np.array([0,0,x[13]])

        geodetic = pm.ecef2geodetic(x[0], x[1], x[2])

        return np.array([
            acc_vector[0],
            acc_vector[1],
            acc_vector[2],
            x[10],
            x[11],
            x[12],
            x[0],
            x[1],
            x[2],
            geodetic[2]
        ])
    
    def fx(self, x: np.ndarray, dt: float):
        '''
        State transition function
        '''
        new_x = np.zeros_like(x)
        orientation = R.from_quat(x[6:10])
        thrust_vec = orientation.apply(self.original_direction) * x[13]
        unit_vector_from_earth_center = x[:3] / np.linalg.norm(x[:3])
        gravity_vec = -unit_vector_from_earth_center * 9.81
        accel_vec = thrust_vec + gravity_vec
        new_x[:3] = x[:3] + x[3:6]*dt + 0.5*accel_vec*dt**2
        new_x[3:6] = x[3:6] + accel_vec*dt
        new_x[6:10] = R.as_quat(R.from_euler('xyz', x[10:13]*dt) * orientation)
        new_x[10:13] = x[10:13]
        new_x[13] = x[13] + x[14]*dt
        new_x[14] = x[14]

        return new_x
    
    def predict_update(self, dt: float, z: np.ndarray, debug_logging: tuple[SummaryWriter, int] = None):
        '''
        debug_logging is a tuple of (SummaryWriter, int) where the int is the current iteration number
        '''
        S = cholesky(self.P + self.Q) 
        n = S.shape[0]

        # columns of W are the sigma points without mean shift
        # should be shape (14, 2*14)
        W = np.concatenate([np.sqrt(2*n) * S, -np.sqrt(2*n) * S], axis=1)
        w_q = self._to_quat(W[6:9])
        
        X = np.vstack([ # sigma points
            W[:6]+self.x[:6,None],
            self._quat_mult(self.x[6:10], w_q),
            W[9:14]+self.x[10:15,None]
        ])

        X = np.apply_along_axis(self.fx, 0, X, dt)

        pred_x = np.hstack([
            np.mean(X[:6], axis=1),
            self._avg_quaternions(X[6:10].T),
            np.mean(X[10:15], axis=1)
        ])

        # begin update step

        a_priori_P = 1/(2*n) * np.sum([
            np.outer(W[:,i], W[:,i].T)
            for i in range(2*n)
        ], axis=0) # using this instead of covariance of actual sigmas

        Z = np.apply_along_axis(self.hx, 0, X)

        z_hat = np.mean(Z, axis=1)
        P_zz = np.cov(Z)

        P_vv = P_zz + self.R

        # the original paper fails to mention this, but this matrix has one less
        # dimension than the state, so to use it for the Kalman gain we need to
        # convert the rotation vectors to quaternions. But, for the covariance
        # matrix, we have to use the rotation vectors.
        P_xz = 1/(2*n) * np.sum([
            np.outer(W[:,i], (Z[:,i] - z_hat))
            for i in range(2*n)
        ], axis=0)


        K = P_xz @ np.linalg.inv(P_vv)

        # this is also not mentioned in the paper, but we can't just add
        # the quaternion updates to the state vector. We have to multiply them.
        delta_x =  K @ (z - z_hat)

        if debug_logging is not None:
            debug_logging[0].add_scalar("z surprise", np.linalg.norm(z - z_hat), debug_logging[1])

        self.x = np.hstack([
            pred_x[:6] + delta_x[:6],
            self._quat_mult(pred_x[6:10], self._to_quat(delta_x[6:9])),
            pred_x[10:] + delta_x[9:]
        ])
        self.P = a_priori_P - K @ P_vv @ K.T


        
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