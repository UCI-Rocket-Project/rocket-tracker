import numpy as np

class LaunchDetector:
    def __init__(self, initial_pixel_position: tuple[float, float], threshold = 5):
        self.initial_pixel_position = np.array(initial_pixel_position)
        self.detected_launch = False
        self.threshold = threshold
        self.launch_time = None
    
    def update(self, pixel_position: tuple[float, float], time: float):
        if np.linalg.norm(np.array(pixel_position) - self.initial_pixel_position) > self.threshold:
            self.detected_launch = True
            self.launch_time = time

    def has_detected_launch(self):
        return self.detected_launch

    def get_launch_time(self) -> float:
        return self.launch_time


