import numpy as np
from .utils import TelemetryData
from .environment import Environment
from .haz31_telescope import HAZ31Telescope
import cv2

import time
from pathlib import Path
from perception.types import Image
import cv2 as cv
from enum import Enum
from shared.types import Pose
import numpy as np
from scipy.spatial.transform import Rotation


class Arducam:
    class ResolutionOption(Enum):
        R1080P = (1920, 1080)
        R720P = (1280, 720)
        R480P = (640, 480)

    class ArducamPose(Enum):
        FRONT = Pose(np.array([0.1, 0, 0]), Rotation.identity())
        DOWN = Pose(
            np.array([0.1, 0, -0.05]), Rotation.from_euler("Y", 90, degrees=True)
        )

    def __init__(
        self,
        log_dir: str | Path | None = None,
        resolution: ResolutionOption = ResolutionOption.R1080P,
        relative_pose: Pose = ArducamPose.FRONT.value,
        flipped=False,  # because of how they're mounted we might have to flip them sometimes.
        video_path="/dev/video0",
    ):
        pipeline = (
            rf"v4l2src device={video_path} io-mode=2 ! "
            rf"image/jpeg,width={resolution.value[0]},height={resolution.value[1]},framerate=30/1 ! "
            r"jpegdec ! "
            r"videoconvert ! "
            r"video/x-raw, format=BGR ! "
            r"appsink drop=true max-buffers=1"
        )
        self._cap = cv.VideoCapture(pipeline)
        self._relative_pose = relative_pose
        self._resolution = resolution
        self._flipped = flipped

    def take_image(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        if self._flipped:
            frame = cv.rotate(frame, cv.ROTATE_180)
        return frame

    def get_focal_length_px(self):
        return 1020 * self._resolution.value[0] / 1920
        # found through checkerboard calibration using Eesh's scripts: https://github.com/InspirationRobotics/RX24-perception/tree/main/camera/dev/calibration
        # these are at ~/camera_calib on the jetson, and for the undistortion one you have to source the venv.


class USBCamEnvironment(Environment):
    def __init__(self):
        super().__init__((0,0,0), (0,0,0), (1920, 1080), 714, 7)

        self.cam=Arducam()
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
        img = self.cam.take_image()
        if img is None:
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
