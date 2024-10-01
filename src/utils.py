import numpy as np
from dataclasses import dataclass
@dataclass
class GroundTruthTrackingData:
    '''
    Dataclass for tracking data from the ground truth simulation
    '''
    pixel_coordinates: tuple[int,int]
    enu_coordinates: np.ndarray
    az_alt: tuple[float,float]

@dataclass
class TelemetryData:
    '''
    Dataclass for telemetry data from the rocket. Right now it excludes the IMU data (accel and gyro)
    '''
    gps_lat: float
    gps_lng: float
    gps_height: float
    v_north: float
    v_east: float
    v_down: float
    accel_x: float
    accel_y: float
    accel_z: float
    time: float

def azi_rot_mat(azi: float):
    '''
    `azi` should be in degrees. Returns a 3x3 rotation matrix.
    Uses a coordinate system where z is up and y is north
    '''
    azi = np.deg2rad(azi)
    return np.array([
        [np.cos(azi),  -np.sin(azi),  0],
        [np.sin(azi),  np.cos(azi),  0],
        [0,  0,  1]
    ])

def alt_rot_mat(alt: float):
    '''
    `alt` should be in degrees. Returns a 3x3 rotation matrix.
    Uses a coordinate system where z is up and y is north
    '''
    alt = np.deg2rad(alt)
    return np.array([
        [1,  0,  0],
        [0,  np.cos(alt),  -np.sin(alt)],
        [0,  np.sin(alt), np.cos(alt)] 
    ])

def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the angle between two vectors in degrees
    '''
    return np.rad2deg(np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))

def aim_direction_to_vec(aim: tuple[float,float]):
    '''
    Converts an aim direction (azimuth, altitude) to a unit vector
    '''
    return alt_rot_mat(aim[1]) @ azi_rot_mat(aim[0]) @ np.array([0,1,0])

def angle_between_aim_directions(aim1: tuple[float,float], aim2: tuple[float,float]):
    '''
    Returns the angle between two aim directions in degrees
    '''
    return angle_between_vectors(aim_direction_to_vec(aim1), aim_direction_to_vec(aim2))

def coord_to_pixel(coord: np.ndarray, camera_pos: np.ndarray, camera_heading: tuple[float,float], camera_res: tuple[int, int], camera_focal_len: float) -> tuple[int,int]:
    '''
    coord and camera_pos are of shape (3,)
    camera_res is (width, height)
    TODO: figure out assumption about where the camera is pointed at 0 azimuth and altitude
    camera focal len needs to be in pixels
    '''
    t_azi, t_alt = camera_heading
    az, alt = np.deg2rad(t_azi), np.deg2rad(t_alt)
    alt_rotation = np.array([
        [1,  0,  0],
        [0,  np.cos(alt),  -np.sin(alt)],
        [0,  np.sin(alt), np.cos(alt)] 
    ])

    az_rotation = np.array([
        [np.cos(az),  -np.sin(az),  0],
        [np.sin(az),  np.cos(az),  0],
        [0,  0,  1]
    ])

    rocket_cam_pos = alt_rotation.T @ az_rotation.T @ (coord - camera_pos)

    w,h = camera_res

    focal_len_pixels = camera_focal_len

    pixel_x = w/2 + focal_len_pixels * rocket_cam_pos[0]/rocket_cam_pos[1] 
    pixel_y = h/2 - focal_len_pixels * rocket_cam_pos[2]/rocket_cam_pos[1] 

    return int(pixel_x), int(pixel_y)
