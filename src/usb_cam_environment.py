import numpy as np
from .utils import TelemetryData
from .environment import Environment
from .haz31_telescope import HAZ31Telescope
import cv2

class USBCamEnvironment(Environment):
    def __init__(self):
        super().__init__((0,0,0), (0,0,0), (1920, 1080), 714, 7)

        self.cap = cv2.VideoCapture(2)
        self.telescope = HAZ31Telescope()
        # 4900 is zero position to line up shaft collar on telescope
        self.focuser_bounds = (0,1)

    def get_telescope_orientation(self) -> tuple[float,float]:
        return self.telescope.read_position() 

    def get_telescope_speed(self) -> tuple[float, float]:
        return self.telescope.read_azi_speed(), self.telescope.read_alt_speed() 

    def move_telescope(self, v_azimuth: float, v_altitude: float):
        self.telescope.slew_rate_azi_alt(-v_azimuth, v_altitude)

    def get_camera_image(self) -> np.ndarray:
        ret, img = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def set_camera_settings(self, gain: int, exposure: int):
        pass


    def move_focuser(self, position: int):
        pass

    def get_focuser_position(self) -> int:
        return 0

    def get_telemetry(self) -> TelemetryData:
        return None

    def get_focuser_bounds(self) -> tuple[int, int]:
        return (self.focuser_bounds[0], self.focuser_bounds[1])
