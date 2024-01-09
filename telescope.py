import serial
from time import sleep

class Telescope:
    def __init__(self, sim=True):
        self.serial_connection = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        # read mount info
        self.serial_connection.write(b':MountInfo#')
        mount_info = self.serial_connection.read(4)
        if mount_info == b'0035':
            # switch to special mode
            sleep(1)
            self.serial_connection.write(b':ZZZ#')
            sleep(1)
            self.serial_connection.write(b':MountInfo#')
            mount_info = self.serial_connection.read(4)
            if mount_info != b'8035':
                raise RuntimeError("Failed to switch to special mode")
        elif mount_info != b'8035':
            raise RuntimeError("Failed to connect to telescope")
        print("Telescope connected")

    def slew_rate_azi_alt(self, azi_rate: float, alt_rate: float):
        '''
        Rates are both in degrees per second
        Max magnitude is 8 for azi, and 6 for alt
        '''
        assert abs(azi_rate) <= 8
        assert abs(alt_rate) <= 6 

        # convert to integer unit of 0.01 arcsec per second
        azi_rate_arcsec = abs(int(azi_rate * 3600 * 100))
        alt_rate_arcsec = abs(int(alt_rate * 3600 * 100))

        azi_sign = '+' if azi_rate >= 0 else '-'
        alt_sign = '+' if alt_rate >= 0 else '-'

        command = f":M2{azi_sign}{azi_rate_arcsec:07d}{alt_sign}{alt_rate_arcsec:07d}#"
        print("Sending command:", command)
        self.serial_connection.write(command.encode())
        
        success = self.serial_connection.read(1) == b'1'

        if not success:
            raise RuntimeError("Slew rate command failed")

    def read_azi_speed(self) -> float:
        '''
        Returns the azimuth slew rate in degrees per second
        '''
        self.serial_connection.write(b":Q0#")
        response = self.serial_connection.read(9)
        un_converted_speed = int(response[1:-1])
        return un_converted_speed / 3600 / 100

    def read_alt_speed(self) -> float:
        '''
        Returns the altitude slew rate in degrees per second
        '''
        self.serial_connection.write(b":Q1#")
        response = self.serial_connection.read(9)
        un_converted_speed = int(response[1:-1])
        return un_converted_speed / 3600 / 100

    def read_position(self) -> tuple[float,float]:
        '''
        Returns the azimuth and altitude in degrees        
        '''

        self.serial_connection.write(b":P2#")        
        raw_response = self.serial_connection.read(3+2*7)
        azi_raw_pos = int(raw_response[1:8])
        alt_raw_pos = int(raw_response[9:16])

        azi_pos = azi_raw_pos / 3600 / 100
        alt_pos = alt_raw_pos / 3600 / 100

        return azi_pos, alt_pos
    
    @property
    def Azimuth(self):
        return self.T.Azimuth

    @property
    def Altitude(self):
        return self.T.Altitude

