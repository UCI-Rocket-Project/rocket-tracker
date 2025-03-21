import cv2
import numpy as np
import time
import logging
import threading
import subprocess

# Default stream: rtsp://192.168.144.25:8554/main.264

# Does not work on my mac but maybe on Jetson?
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Untested but could work if we build opencv with gstreamer:
# This will automatically grab latest frame
# cv2.VideoCapture("rtspsrc location=rtsp://... ! decodebin ! videoconvert ! video/x-raw,framerate=30/1 ! appsink drop=true sync=false", cv2.CAP_GSTREAMER)

# With the current method, we can grab the latest frame, but it will be ~0.6 seconds behind realtime to due opencv being dogshit. (you can see this by running the test_continuous function)


class SIYISTREAM:
    def __init__(
        self,
        server_ip: str = "192.168.144.25",
        port: int = 8554,
        name: str = "main.264",
        debug: bool = False,
    ):
        """

        Params
        --
        - server_ip [str] IP address of the camera
        - port: [int] UDP port of the camera
        - name: [str] name of the stream
        """
        self._debug = debug  # print debug messages
        if self._debug:
            d_level = logging.DEBUG
        else:
            d_level = logging.INFO
        LOG_FORMAT = (
            " [%(levelname)s] %(asctime)s [SIYISDK::%(funcName)s] :\t%(message)s"
        )
        logging.basicConfig(format=LOG_FORMAT, level=d_level)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("Initializing SIYISTREAM")
        if self._debug:
            self._logger.info("Debug mode")
        self._server_ip = server_ip
        self._port = port
        self._name = name
        self._stream_link = (
            "rtsp://" + self._server_ip + ":" + str(self._port) + "/" + self._name
        )
        self._logger.info("Stream link: {}".format(self._stream_link))
        self._stream = None

        # Due to the queue like nature of opencv (really fricking stupid), we need to use a bufferless video capture
        self.lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self._read_stream)
        self.capture_thread.daemon = True
        self.bad_count = 0

    # grab frames as soon as they are available
    def _read_stream(self):
        while True:
            with self.lock:
                ret = self._stream.grab()
                # self._logger.debug("Grabbed frame")
                pass
            if not ret:
                break
            # Delay so the thread lock isn't hogged
            time.sleep(1 / 35)  # slightly above 30fps

    def connect(self):
        """
        Connect to the camera
        """
        if self._stream is not None:
            self._logger.warning("Already connected to camera")
            return

        # nvvidconv is only available on the jetson. To test the camera on other computers
        # we need to use plain old videoconvert instead.
        result = subprocess.run(
            ["gst-inspect-1.0", "nvvidconv"],
            env={"PAGER": "cat"},  # gst-inspect opens a pager without this
            capture_output=True,
            text=True,
        )

        # Check if the return code is zero and print the output
        video_convert_plugin = "nvvidconv" if result.returncode == 0 else "videoconvert"

        gst_pipeline = (
            f"rtspsrc location={self._stream_link} latency=10 ! "
            "rtpjitterbuffer latency=10 ! "
            "decodebin ! "
            f"{video_convert_plugin} ! "
            "video/x-raw, format=BGRx ! "
            "appsink drop=true"
        )

        print(gst_pipeline)
        self._stream = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        # Check if camera is connected
        if not self._stream.isOpened():
            self._logger.error("Unable to connect to camera")
            self._stream = None
            return
        # Start capture thread
        self.capture_thread.start()
        self._logger.info("Connected to camera")
        # Let the buffer fill
        time.sleep(2)
        return True

    def disconnect(self):
        """
        Disconnect from the camera
        """
        if self._stream is None:
            self._logger.warning("Already disconnected from camera")
            return
        self._stream.release()
        self._stream = None
        self._logger.info("Disconnected from camera")
        return

    def get_frame(self) -> np.ndarray | None:
        """
        Get a frame from the stream
        """
        if self._stream is None:
            self._logger.warning("Not connected to camera")
            return
        ret = False
        while not ret:
            # self._logger.debug("Waiting for lock")
            with self.lock:
                # self._logger.debug("Lock acquired")
                ret, frame = self._stream.retrieve()
                if ret:
                    # self._logger.debug("Frame read")
                    if self.bad_count > 0:
                        self._logger.info(f"Reset bad count from {self.bad_count}")
                        self.bad_count = 0
                    pass
                else:
                    self._logger.warning(
                        f"Unable to read frame ({self.bad_count} in a row)"
                    )
                    return None
                    self.bad_count += 1
        if frame.ndim < 3:
            frame = np.stack([frame, frame, frame], axis=2)
        return frame[:, :, :3]
