from src.environment import Environment
from tqdm import tqdm
from src.utils import TelemetryData, coord_to_pixel
from simulation.sim_telescope import SimTelescope
from simulation.rocket import Rocket
from torch.utils.tensorboard import SummaryWriter
from src.component_algos.img_tracking import BaseImageTracker
from src.tracker import Tracker
from scipy.spatial.transform import Rotation as R
from time import time, sleep
import pymap3d as pm
import numpy as np
import os
import shutil


class HeadlessSim:
    def __init__(self):
        self.camera_res = (1920//2, 1080)
        self.camera_fov = 0.9
        focal_len_pixels = self.camera_res[0]/(2*np.tan(np.deg2rad(self.camera_fov/2)))
        # fov should be around 0.9 degrees IRL
        # self.camera_fov = np.rad2deg(2*np.arctan(self.camera_res[0]/(2*focal_len_pixels)))
        self.cam_focal_len_pixels = focal_len_pixels
        self.telescope = SimTelescope(-69.62, 0)
        # camera is at (0,0,0) but that's implicit

        self.pad_geodetic_pos = np.array([35.347104, -117.808953, 620])
        self.cam_geodetic_location = np.array([35.34222222, -117.82500000, 620])

        self.logger = SummaryWriter("runs/headless_simulation/true")
        self.launch_time = 2
        self.rocket = Rocket(self.pad_geodetic_pos, self.launch_time)
        self.time = 0
        self.sim_duration_seconds = 30
        self.dt = 1/30
        self.last_time = None
        self.telescope = SimTelescope(-69.62, 0)
        self.focuser_position = 0
        self.focuser_bounds = (0, 60)
        class HeadlessSimEnvironment(Environment):
            def __init__(env_self, pad_pos_gps: tuple[float, float, float], cam_pos_gps: tuple[float, float, float], camera_resolution: tuple[int, int], camera_focal_len_pixels: float, cam_fstop: float):
                super().__init__(pad_pos_gps, cam_pos_gps, camera_resolution, camera_focal_len_pixels, cam_fstop) 
    
            def get_telescope_orientation(env_self) -> tuple[float,float]:
                '''
                Return the azimuth and altitude of the telescope in degrees
                '''
                return self.telescope.read_position()

            def get_telescope_speed(env_self) -> tuple[float,float]:
                return self.telescope.read_speed()

            def move_telescope(env_self, v_azimuth: float, v_altitude: float):
                return self.telescope.slew_rate_azi_alt(v_azimuth, v_altitude)

            def get_camera_image(env_self) -> np.ndarray:
                return np.zeros([1920, 1080], dtype=np.uint8)

            def move_focuser(env_self, position: int):
                self.focuser_position = position

            def get_focuser_bounds(env_self) -> tuple[int,int]:
                return self.focuser_bounds

            def get_focuser_position(env_self) -> int:
                return self.focuser_position

            def get_telemetry(env_self) -> TelemetryData:
                return self.rocket.get_telemetry(self.time)
        
        class MockImageTracker(BaseImageTracker):
            def estimate_pos(tracker_self, img: np.ndarray) -> tuple[int, int, int, int]:
                '''
                Returns bounding box [center_x, center_y, width, height] of the highest confidence detection
                '''

                rocket_ecef_pos = self.rocket.get_position_ecef(self.time - self.launch_time)
                rocket_enu_camera_frame_pos = pm.ecef2enu(*rocket_ecef_pos, *self.environment.get_cam_pos_gps())                

                # reference points are the top and bottom of the rocket plus or minus the rocket radius
                # this will assume that the rocket is pointed in the direction of its velocity vector
                rocket_radius = 0.9
                rocket_height = 6.5
                rocket_orientation = self.rocket.get_orientation(self.time - self.launch_time)
                vel_direction = rocket_orientation.apply([0,0,1])

                top_point = rocket_enu_camera_frame_pos + rocket_height/2 * vel_direction
                bottom_point = rocket_enu_camera_frame_pos - rocket_height/2 * vel_direction

                orthogonal_direction = np.cross(vel_direction, np.array([1,0,0])) # arbitrary vector that is orthogonal to the velocity vector
                orthogonal_direction_2 = np.cross(vel_direction, orthogonal_direction) # second orthogonal vector, also orthogonal to the first

                top_4_points = [
                    top_point + rocket_radius * orthogonal_direction,
                    top_point - rocket_radius * orthogonal_direction,
                    top_point + rocket_radius * orthogonal_direction_2,
                    top_point - rocket_radius * orthogonal_direction_2
                ]

                bottom_4_points = [
                    bottom_point + rocket_radius * orthogonal_direction,
                    bottom_point - rocket_radius * orthogonal_direction,
                    bottom_point + rocket_radius * orthogonal_direction_2,
                    bottom_point - rocket_radius * orthogonal_direction_2
                ]

                all_points = np.array(list.__add__(top_4_points,bottom_4_points)) # 8 points, using __add__ to make it clear it's not element-wise

                pixel_coords = np.array([
                    coord_to_pixel(
                        point, np.zeros(3),
                        self.environment.get_telescope_orientation(),
                        self.camera_res, self.cam_focal_len_pixels
                    ) 
                    for point in all_points
                ])

                min_x, min_y = np.min(pixel_coords, axis=0)
                max_x, max_y = np.max(pixel_coords, axis=0)
                center_x = int((min_x+max_x)/2)
                center_y = int((min_y+max_y)/2)
                return center_x, center_y, int(max_x-min_x), int(max_y-min_y)
            
        self.img_tracker = MockImageTracker()
                    
        self.environment = HeadlessSimEnvironment(
            self.pad_geodetic_pos,
            self.cam_geodetic_location,
            self.camera_res,
            self.cam_focal_len_pixels,
            cam_fstop = 7,
        )
        self.tracker = Tracker(
            self.environment, 
            SummaryWriter('runs/headless_simulation/pred'),
            self.img_tracker
        )

        self.tracker.start_tracking(self.environment.get_telescope_orientation())

    def _pixel_pos_to_az_alt(self, pixel_pos: np.ndarray) -> tuple[float,float]:
        az = -np.arctan2(pixel_pos[0] - self.camera_res[0] / 2, self.cam_focal_len_pixels)
        alt = -np.arctan2(pixel_pos[1] - self.camera_res[1] / 2, self.cam_focal_len_pixels)
        pixel_rot = R.from_euler("ZY", [az, alt], degrees=False)
        initial_rot = R.from_euler("ZY", self.environment.get_telescope_orientation(), degrees=True)

        final_rot = initial_rot * pixel_rot
        az, alt = final_rot.as_euler("ZYX", degrees=True)[:2]
        return az, alt
        

    def step(self, step_time: float):
        '''
        step_time is time since simulation start in seconds
        '''
        self.time = step_time
        self.telescope.step(step_time)
        self.tracker.update_tracking(
            self.environment.get_camera_image(),
            self.environment.get_telemetry(),
            step_time,
            control_scope = True
        )

        x,y,z = self.rocket.get_position(step_time)
        rocket_pos_ecef = self.rocket.get_position_ecef(step_time)
        rocket_vel = self.rocket.get_velocity(step_time)
        rocket_accel = self.rocket.get_acceleration(step_time)
        self.logger.add_scalar("enu position/x", x, step_time*100)
        self.logger.add_scalar("enu position/y", y, step_time*100)
        self.logger.add_scalar("enu position/z", z, step_time*100)

        dist_to_camera = np.linalg.norm(rocket_pos_ecef - np.array(pm.geodetic2ecef(*self.cam_geodetic_location)))
        self.logger.add_scalar('mount/distance', dist_to_camera, step_time*100)

        self.logger.add_scalar("enu velocity/x", rocket_vel[0], step_time*100)
        self.logger.add_scalar("enu velocity/y", rocket_vel[1], step_time*100)
        self.logger.add_scalar("enu velocity/z", rocket_vel[2], step_time*100)

        self.logger.add_scalar("enu accel/x", rocket_accel[0], step_time*100)
        self.logger.add_scalar("enu accel/y", rocket_accel[1], step_time*100)
        self.logger.add_scalar("enu accel/z", rocket_accel[2], step_time*100)
        bbox = self.img_tracker.estimate_pos(None)
        pixel_x, pixel_y = bbox[:2]
        self.logger.add_scalar("pixel position/x", pixel_x, step_time*100)
        self.logger.add_scalar("pixel position/y", pixel_y, step_time*100)
        self.logger.add_scalar("pixel position/bbox_w", bbox[2], step_time*100)
        self.logger.add_scalar("pixel position/bbox_h", bbox[3], step_time*100)
        self.logger.add_scalar("pixel position/bbox_diag", np.linalg.norm(bbox[2:]), step_time*100)

        az_new, alt_new = self._pixel_pos_to_az_alt(bbox[:2])
        self.logger.add_scalar("bearing/azimuth", az_new, step_time*100)
        self.logger.add_scalar("bearing/altitude", alt_new, step_time*100)

        # az_derivative = (az_new-az_old)/(task.time-self.prev_rocket_observation_time)
        # alt_derivative = (alt_new-alt_old)/(task.time-self.prev_rocket_observation_time)
        launched = int(step_time>=self.launch_time)

        geodetic_pos = pm.ecef2geodetic(*rocket_pos_ecef)
        self.logger.add_scalar("telemetry/lat", geodetic_pos[0], step_time*100)
        self.logger.add_scalar("telemetry/lng", geodetic_pos[1], step_time*100)
        self.logger.add_scalar("telemetry/alt", geodetic_pos[2], step_time*100)

        self.logger.add_scalar("launched", launched, step_time*100)

    
    def run(self):
        self.start_time = time()
        def generator():
            while True:
                wall_time = time()
                if self.last_time is None:
                    yield wall_time
                if wall_time < self.last_time + self.dt:
                    sleep(self.dt + self.last_time - wall_time)
                yield wall_time

        for wall_time in tqdm(generator()):
            self.step(wall_time - self.start_time)
            self.last_time = wall_time
            if wall_time-self.start_time > self.sim_duration_seconds:
                quit()

if __name__ == '__main__':
    if os.path.exists("runs/headless_simulation"):
        shutil.rmtree("runs/headless_simulation")
    os.makedirs("runs/headless_simulation")
    sim = HeadlessSim()
    sim.run()