from torch.utils.tensorboard import SummaryWriter


class SimTelescope:
    '''
    Simulated telescope mount that should have the same interface as the real telescope class (telescope.Telescope)
    '''
    def __init__(self, azimuth=0, altitude=0, logger: SummaryWriter = None):
        self._azimuth = azimuth 
        self._altitude = altitude 
        self._azimuth_rate = 0
        self._altitude_rate = 0
        self._last_updated_time = 0
        self.is_fake = True
        self.logger = logger

    def slew_rate_azi_alt(self, azi_rate: float, alt_rate: float):
        '''
        Slew the telescope at the given rates in degrees per second
        time (in seconds) is used to calculate the change in position
        '''
        self._azimuth_rate = azi_rate
        self._altitude_rate = alt_rate

    def step(self, time):
        self._altitude+= self._altitude_rate*(time-self._last_updated_time)
        self._azimuth+= self._azimuth_rate*(time-self._last_updated_time)
        self._last_updated_time = time
        self.logger.add_scalar("mount/azimuth", self._azimuth, time)
        self.logger.add_scalar("mount/altitude", self._altitude, time)
        self.logger.add_scalar("mount/azimuth_rate", self._azimuth_rate, time)
        self.logger.add_scalar("mount/altitude_rate", self._altitude_rate, time)

    def read_position(self):
        return self._azimuth, self._altitude
    
    def read_speed(self) -> tuple[float,float]:
        return (self._azimuth_rate, self._altitude_rate)
