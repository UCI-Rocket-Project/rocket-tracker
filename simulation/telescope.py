class Telescope:
    def __init__(self, azimuth=0, altitude=0):
        self.Azimuth = azimuth 
        self.Altitude = altitude 
        self._azimuth_rate = 0
        self._altitude_rate = 0
        self._last_updated_time = 0
    
    def slewAzimuthRate(self, rate, time):
        self.step(time)
        self._azimuth_rate = rate

    def slewAltitudeRate(self, rate, time):
        self.step(time)
        self._altitude_rate = rate
    
    def step(self, time):
        self.Altitude+= self._altitude_rate*(time-self._last_updated_time)
        self.Azimuth+= self._azimuth_rate*(time-self._last_updated_time)
        self._last_updated_time = time
