
class Telescope:
    def __init__(self):
        pass

    def slew_rate_azi_alt(self, azi_rate: float, alt_rate: float):
        pass

    def read_speed(self) -> tuple[float, float]:
        pass

    def read_position(self) -> tuple[float,float]:
        pass

    def stop(self):
        pass
