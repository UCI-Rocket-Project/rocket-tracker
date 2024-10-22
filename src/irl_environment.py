import numpy as np
from .utils import TelemetryData
from .environment import Environment
from zwo_asi import ASICamera
from zwo_eaf import EAF, getNumEAFs, getEAFID
from .haz31_telescope import HAZ31Telescope

class IRLEnvironment(Environment):
    def __init__(self):
        super().__init__((0,0,0), (0,0,0), (1920, 1080), 714, 7)

        eaf_count = getNumEAFs()

        print(f"Found {eaf_count} EAFs")
        self.focuser = EAF(getEAFID(0))
        self.cam = ASICamera(0,0,1000*1/120)

        self.telescope = HAZ31Telescope()
        # 4900 is zero position to line up shaft collar on telescope
        self.focuser_bounds = (4900, 24000)

    def get_telescope_orientation(self) -> tuple[float,float]:
        return self.telescope.read_position() 

    def get_telescope_speed(self) -> tuple[float, float]:
        return self.telescope.read_azi_speed(), self.telescope.read_alt_speed() 

    def move_telescope(self, v_azimuth: float, v_altitude: float):
        print(v_azimuth, v_altitude)
        self.telescope.slew_rate_azi_alt(-v_azimuth, v_altitude)

    def get_camera_image(self) -> np.ndarray:
        img = self.cam.get_frame()
        return img.copy()

    def set_camera_settings(self, gain: int, exposure: int):
        del self.cam
        print(gain, exposure)
        self.cam = ASICamera(0,gain,exposure)


    def move_focuser(self, position: int):
        assert isinstance(position, int)
        if position not in range(self.focuser_bounds[0], self.focuser_bounds[1]):
            raise ValueError(f"Position {position} is out of bounds")
        self.focuser.move_to(position)

    def get_focuser_position(self) -> int:
        return self.focuser.get_position()

    def get_telemetry(self) -> TelemetryData:
        return None

    def get_focuser_bounds(self) -> tuple[int, int]:
        return (self.focuser_bounds[0], self.focuser_bounds[1])
