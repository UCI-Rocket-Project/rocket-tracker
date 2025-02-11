import numpy as np
from .utils import TelemetryData
from .environment import Environment
from .zr10 import SIYISDK, SIYISTREAM

class ZR10Environment(Environment):
    def __init__(self):
        super().__init__((0,0,0), (0,0,0), (1920, 1080), 714, 7)

        self.siyi_sdk = SIYISDK()
        self.siyi_stream = SIYISTREAM()

        self.siyi_stream.connect()
        self.siyi_sdk.connect()
        self.siyi_sdk.setAbsoluteZoom(10)

    def get_telescope_orientation(self) -> tuple[float,float]:
        return self.siyi_sdk.getAttitude()[:2]

    def get_telescope_speed(self) -> tuple[float, float]:
        return self.siyi_sdk.getAttitudeSpeed()[:2]

    def move_telescope(self, v_azimuth: float, v_altitude: float):
        self.siyi_sdk.requestGimbalSpeed(-int(v_azimuth*100/8), int(v_altitude*100/6))

    def get_camera_image(self) -> np.ndarray:
        img = self.siyi_stream.get_frame()
        img = np.ascontiguousarray(img, dtype=np.uint8)
        return img

    def set_camera_settings(self, gain: int, exposure: int):
        pass

    def move_focuser(self, position: int):
        self.siyi_sdk.doAutoFocus()

    def get_focuser_position(self) -> int:
        return 0

    def get_telemetry(self) -> TelemetryData:
        return None

    def get_focuser_bounds(self) -> tuple[int, int]:
        return (0, 0)

