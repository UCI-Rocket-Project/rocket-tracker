from alpaca.telescope import TelescopeAxes, Telescope as AlpacaTelescope

class Telescope:
    def __init__(self, sim=True):
        address = 'localhost:32323' if sim else '127.0.0.1:11111'
        self.T = AlpacaTelescope(address, 0)
        self.T.Connected = True
        for rate in self.T.AxisRates(TelescopeAxes.axisPrimary):
            print(rate.Maximum, rate.Minimum)
    
    def slewAzimuthRate(self, rate: float):
        self.T.MoveAxis(TelescopeAxes.axisPrimary, rate)

    def slewAltitudeRate(self, rate: float):
        self.T.MoveAxis(TelescopeAxes.axisSecondary, rate)
    
    @property
    def Azimuth(self):
        return self.T.Azimuth

    @property
    def Altitude(self):
        return self.T.Altitude

