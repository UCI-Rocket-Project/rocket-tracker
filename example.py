import time
from alpaca.telescope import Telescope      # Multiple Classes including Enumerations
from alpaca.exceptions import InvalidOperationException

T = Telescope('localhost:32323', 0) # Local Omni Simulator
try:
    T.Connected = True
    print(f'Connected to {T.Name}')
    print(T.Description)
    T.Tracking = True               # Needed for slewing (see below)
    print('Starting slew...')
    T.SlewToCoordinatesAsync(T.SiderealTime + 2, 50)    # 2 hrs east of meridian
    while(T.Slewing):
        time.sleep(5)               # What do a few seconds matter?
    print('... slew completed successfully.')
    print(f'RA={T.RightAscension} DE={T.Declination}')
    print('Turning off tracking then attempting to slew...')
    T.Tracking = False
    T.SlewToCoordinatesAsync(T.SiderealTime + 2, 55)    # 5 deg slew N
    # This will fail for tracking being off
    print("... you won't get here!")
except InvalidOperationException as e:              # Should catch specific InvalidOperationException
    print(f'Slew failed: {str(e)}')
finally:                            # Assure that you disconnect
    print("Disconnecting...")
    T.Connected = False