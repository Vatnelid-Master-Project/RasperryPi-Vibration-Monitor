import time

import Adafruit_ADS1x15

GAIN = 16
adc = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)

def app():
    print(adc.read_adc(0, gain=GAIN))
    time.sleep(2)
