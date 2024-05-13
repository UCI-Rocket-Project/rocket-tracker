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

        x_dim = 7
        z_dim = 3
        self.x = np.empty(x_dim) # state vector
        self.x[:4] = R.identity().as_quat()
        self.x[4:] = 0
        

        # covariance matrices have 1 less dimension because the quaternion orientation
        # is 4 variables, but only 3 degrees of freedom
        self.P = np.eye(x_dim - 1) # state covariance matrix
        process_noise_covariances = 1e-3 * np.array([
            0.1, 0.1, 0.1, # orientation
            0.1, 0.1, 0.1, # angular velocity
        ])
        self.Q = np.diag(process_noise_covariances) # process noise covariance matrix

        self.R = 1e-3 * np.eye(z_dim) # measurement noise covariance matrix


    def hx(self, x: np.ndarray):
        '''
        Measurement function
        '''

        return np.array([
            x[4],
            x[5],
            x[6],
        ])
    
    def fx(self, x: np.ndarray, dt: float):
        '''
        State transition function
        '''
        new_x = np.empty_like(x)
        orientation = R.from_quat(x[0:4])
        new_x[0:4] = R.as_quat(R.from_euler('xyz', x[4:]*dt) * orientation)
        new_x[4:6] = x[4:6]

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
        w_q = self._to_quat(W[:4])
        
        X = np.vstack([ # sigma points
            self._quat_mult(self.x[:4], w_q),
            W[3:]+self.x[4:,None]
        ])

        X = np.apply_along_axis(self.fx, 0, X, dt)

        pred_x = np.hstack([
            self._avg_quaternions(X[:4].T),
            np.mean(X[4:], axis=1)
        ]).astype(np.float64)

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
            self._quat_mult(pred_x[:4], self._to_quat(delta_x[:3])),
            pred_x[4:] + delta_x[3:]
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