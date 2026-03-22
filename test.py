import Adafruit_ADS1x15
import time
adc = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)

def test():
	while True:
		print(adc.read_adc(0, gain=16))
		time.sleep(0.1)

test()
