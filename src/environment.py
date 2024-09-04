import numpy as np
from .utils import TelemetryData

class Environment:
    def __init__(self, 
                pad_pos_gps: tuple[float,float,float], 
                cam_pos_gps: tuple[float, float, float],
                camera_resolution: tuple[int,int],
                camera_focal_len_pixels: float,
                cam_fstop: float
        ):
        '''
        Takes rocket initial distance from mount in meters
        '''
        self._pad_pos_gps = pad_pos_gps
        self._cam_pos_gps = cam_pos_gps
        self._camera_resolution = camera_resolution
        self._camera_focal_len_pixels = camera_focal_len_pixels
        self._cam_fstop = cam_fstop

    def get_pad_pos_gps(self) -> tuple[float,float,float]:
        '''
        Return the position of the launch pad in GPS coordinates (lat,lng,alt)
        '''
        return self._pad_pos_gps

    def get_cam_pos_gps(self) -> tuple[float,float,float]:
        '''
        Return the position of the camera mount in GPS coordinates (lat,lng,alt)
        '''
        return self._cam_pos_gps

    def get_camera_resolution(self) -> tuple[int,int]:
        return self._camera_resolution

    def get_focal_length_pixels(self) -> float:
        return self._camera_focal_len_pixels

    def get_telescope_orientation(self) -> tuple[float,float]:
        '''
        Return the azimuth and altitude of the telescope in degrees
        '''
        pass

    def get_telescope_speed(self) -> tuple[float,float]:
        pass

    def move_telescope(self, v_azimuth: float, v_altitude: float):
        pass

    def get_camera_image(self) -> np.ndarray:
        pass

    def set_camera_settings(self, gain: int, exposure: float):
        pass

    def move_focuser(self, position: int):
        pass

    def get_focuser_bounds(self) -> tuple[int,int]:
        pass

    def get_focuser_position(self) -> int:
        pass

    def get_telemetry(self) -> TelemetryData:
        pass

    @property
    def cam_fstop(self) -> float:
        return self._cam_fstop