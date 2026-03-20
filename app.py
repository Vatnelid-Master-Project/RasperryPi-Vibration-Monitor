import queue
import threading
import time

import Adafruit_ADS1x15

from receive import process_chunk

GAIN = 16
adc = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)

sample_queue = queue.Queue(maxsize=400)
stop_event = threading.Event()
segment_length = 10000
interval = 5

def assemble():
    '''
    Assemble sensor values into an array
    :return:
    '''
    print('Assembling worker started...')
    result = []
    start = time.time() * 1000
    next_sample = start
    i = 0
    while (time.time() * 1000 - start) <= segment_length:
        now = time.time() * 1000
        if (now - next_sample) >= 0:
            next_sample += interval
            try:
                val = adc.read_adc(0, gain=GAIN, data_rate=860)
                result.append(val)
            except Exception as e:
                print('Exception: ' + str(e))
                break
            i += 1
        else:
            time.sleep(0.001)
        if i >= segment_length:
            break

    try:
        print("Putting the list to the queue...")
        sample_queue.put(result)
    except queue.Full:
        print("Queue full")



def app():
    buf = []
    print('App Worker Started...')
    chunk_size = 2000
    while not stop_event.is_set():
        try:
            print('Trying to access the queue...')
            v = sample_queue.get(timeout=0.5)
        except queue.Empty:
            print('Queue empty')
            continue
        print('Appending to buffer...')
        buf.append(v)
        print(len(buf))
        if len(buf) >= chunk_size:
            chunk = buf[:chunk_size]
            del buf[:chunk_size]  # keep extra samples if they arrived fast

            # process chunk (make df, spectrogram, etc.)
            # IMPORTANT: processing here, not in ws callback
            process_chunk(chunk)

worker = threading.Thread(target=assemble, daemon=True)
app_worker = threading.Thread(target=app, daemon=True)
worker.start()
app_worker.start()

worker.join()
app_worker.join()