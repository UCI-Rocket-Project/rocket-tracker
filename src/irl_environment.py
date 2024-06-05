import numpy as np
from .utils import TelemetryData
from environment import Environment
from zwo_asi import ASICamera
from zwo_eaf import EAF, getEAFID
from haz31_telescope import HAZ31Telescope

class IRLEnvironment(Environment):
    def __init__(self):
        self.cam = ASICamera(0,150,1/30)
        self.focuser = EAF(getEAFID(0))
        self.telescope = HAZ31Telescope()
        self.focuser_bounds = (0, self.focuser.get_max_position())

    def get_telescope_orientation(self) -> tuple[float,float]:
        return self.telescope.read_position() 

    def get_telescope_speed(self) -> tuple[float, float]:
        return self.telescope.read_azi_speed(), self.telescope.read_alt_speed() 

    def move_telescope(self, v_azimuth: float, v_altitude: float):
        self.telescope.slew_rate_azi_alt(v_azimuth, v_altitude)

    def get_camera_image(self) -> np.ndarray:
        return self.cam.get_frame()

    def set_camera_settings(self, gain: int, exposure: float):
        self.cam = ASICamera(0,gain,exposure)

    def get_camera_resolution(self) -> tuple[int,int]:
        return (1920, 1080)

    def get_focal_length(self) -> float:
        print("WARNING: focal length returning fake value rn")
        return 420.69

    def move_focuser(self, position: int):
        if position not in range(self.focuser_bounds[0], self.focuser_bounds[1]):
            raise ValueError(f"Position {position} is out of bounds")
        self.focuser.move_to(position)

    def get_focuser_position(self) -> int:
        return self.focuser.get_position()

    def get_telemetry(self) -> TelemetryData:
        return None
