from alpaca.telescope import TelescopeAxes, Telescope as AlpacaTelescope

class Telescope:
    def __init__(self):
        self.T = AlpacaTelescope('127.0.0.11111', 0)
        self.T.EquatorialSystem
    
    def slewAzimuthRate(self, rate):
        self.T.MoveAxis(TelescopeAxes.axisPrimary, rate)

    def slewAltitudeRate(self, rate, time):
        self.T.MoveAxis(TelescopeAxes.axisSecondary, rate)
    
    @property
    def Azimuth(self):
        return self.T.Azimuth

    @property
    def Altitude(self):
        return self.T.Altitude

