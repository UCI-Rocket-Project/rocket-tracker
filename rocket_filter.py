import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky
import pymap3d as pm

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

        x_dim = 15
        z_dim = 10
        self.x = np.zeros(x_dim) # state vector
        self.x[0:3] = pm.geodetic2ecef(*pad_geodetic_location)
        self.x[6] = 1 # quaternion orientation

        # covariance matrices have 1 less dimension because the quaternion orientation
        # is 4 variables, but only 3 degrees of freedom
        self.P = np.eye(x_dim - 1) # state covariance matrix
        self.Q = np.eye(x_dim - 1) # process noise covariance matrix

        self.R = np.eye(z_dim) # measurement noise covariance matrix


    def hx(self, x: np.ndarray):
        '''
        Measurement function
        '''

        vec_to_earth_center = - x[:3] / np.linalg.norm(x[:3])
        acc_vector = np.array([0,0,x[13]]) + vec_to_earth_center * 9.81

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
        thrust_vec = orientation.apply([0,0,1]) * x[13]
        new_x[:3] = x[:3] + x[3:6]*dt + 0.5*x[10:13]*dt**2
        new_x[3:6] = x[3:6] + thrust_vec*dt
        new_x[6:10] = R.as_quat(R.from_euler('xyz', x[10:13]*dt) * orientation)
        new_x[10:13] = x[10:13]
        new_x[13] = x[13] + x[14]*dt
        new_x[14] = x[14]

        return new_x
    
    def predict_update(self, dt: float, z: np.ndarray):
        S = cholesky(self.P + self.Q) 
        n = S.shape[0]

        # columns of W are the sigma points without mean shift
        # should be shape (14, 2*14)
        W = np.concatenate([np.sqrt(2*n) * S, -np.sqrt(2*n) * S], axis=1)
        w_q = self._to_quat(W[6:9])
        
        X = np.vstack([ # sigma points
            W[:6]+np.mean(self.x[:6]),
            self._quat_mult(self.x[6:10], w_q),
            W[9:14]+np.mean(self.x[10:15])
        ])

        X = np.apply_along_axis(self.fx, 0, X, dt)

        self.x = np.hstack([
            np.mean(X[:6], axis=1),
            self._avg_quaternions(X[6:10].T),
            np.mean(X[10:15], axis=1)
        ])

        # begin update step

        a_priori_P = 1/(2*n) * np.sum([
            W[:,i] @ W[:,i].T
            for i in range(2*n)
        ]) # using this instead of covariance of actual sigmas

        Z = np.apply_along_axis(self.hx, 0, X)

        z_hat = np.mean(Z, axis=1)
        P_zz = np.cov(Z)

        P_vv = P_zz + self.R

        # the original paper fails to mention this, but this matrix has one less
        # dimension than the state, so to use it for the Kalman gain we need to
        # convert the rotation vectors to quaternions. But, for the covariance
        # matrix, we have to use the rotation vectors.
        P_xz_w = 1/(2*n) * np.sum([
            np.outer(W[:,i], (Z[:,i] - z_hat))
            for i in range(2*n)
        ], axis=0)

        W_with_quats = np.vstack([W[:6], w_q, W[9:14]])

        P_xz = 1/(2*n) * np.sum([
            np.outer(W_with_quats[:, i], (Z[:,i] - z_hat))
            for i in range(2*n)
        ], axis=0)

        K = P_xz @ np.linalg.inv(P_vv)
        K_w = P_xz_w @ np.linalg.inv(P_vv)

        self.x = self.x + K @ (z - z_hat)
        self.P = a_priori_P - K_w @ P_vv @ K_w.T


        
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
        e_w = w_q / alpha_w[np.newaxis,:] # some of these will become NaN but they'll be multiplied by 0
        # set all NaNs to 0
        e_w[np.isnan(e_w)] = 0
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