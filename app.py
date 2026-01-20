import queue
import threading
import time
from io import BytesIO
from typing import Callable

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import torch
import requests
import serial
from fastai.vision.all import *
from matplotlib import pyplot as plt
from scipy.signal import ShortTimeFFT, butter, filtfilt
from scipy.signal.windows import hamming

load_dotenv()

URL = os.getenv("URL")

def read_sensor(ser : serial.Serial) -> float:
    try:
        line = ser.readline()
        value = line.decode('utf-8').strip()
        return float(value)
    except IOError:
        print("Error reading sensor")
    return 0.0

def mock_sensor_values(df: pd.DataFrame, index : int):
    return df["value"][index]

def collect_data(
        df_queue: "queue.Queue[pd.DataFrame]",
        stop_event: threading.Event,
):
    csv_file = "./mock.csv"
    csv = pd.read_csv(csv_file)
    df = pd.DataFrame(csv)
    print("Dataframe created...")
    freq = 200
    sec = 10
    dt = 1.0 / freq
    chunk_n = int(freq * sec)
    # collect for 10 seconds
    while not stop_event.is_set():
        i = 0
        t_next = time.monotonic()

        rows = []

        for _ in range(chunk_n):
            if stop_event.is_set():
                break

            rows.append({
                "value": mock_sensor_values(df, i),
            })

            t_next += dt

            sleep_for = t_next - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

            else:
                pass

            i += 1

        # Save the sensor values to a dataframe and put the dataframe
        try:
            print("Dataframe added to queue...")
            df_queue.put(pd.DataFrame(rows), timeout=1)
        except queue.Full:
            pass

def __highpass_filter(x, fs, cutoff=0.5):
    b, a = butter(4, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, x)

def __create_8_channel(data, sampling_fq = 200, nperseg = 64, dynamic_range_db = 50, band_max_hz = None):
    w = hamming(nperseg)
    sft = ShortTimeFFT(w, hop=int(nperseg*0.25), fs=sampling_fq, scale_to='psd')
    sxx = sft.spectrogram(__highpass_filter(data["value"].to_numpy(), sampling_fq))

    sxx_db = 10 * np.log10(sxx + 1e-12)
    vmax = np.percentile(sxx_db, 95)
    vmin = vmax - dynamic_range_db
    sxx_db = np.clip(sxx_db, vmin, vmax)

    f = np.linspace(0, sampling_fq / 2, sxx_db.shape[0])
    if band_max_hz is not None:
        keep = f <= band_max_hz
        sxx_db = sxx_db[keep, :]

    return sxx_db.astype(np.float32)

def save_pyplot(spec, fs = 200, nperseg=64, band_max_hz=80):
    buf = BytesIO()

    global_max = max(s.max() for s in spec)
    global_min = min(s.min() for s in spec)

    n_freq_bins = spec.shape[0]
    freqs = np.linspace(0, fs / 2, n_freq_bins)
    keep = freqs <= band_max_hz

    spec = spec[keep, :]
    freqs = freqs[keep]

    t_min, t_max = 0, 10
    plt.figure(figsize=(8, 4))
    plt.imshow(
        spec,
        vmax=global_max,
        vmin=global_min,
        origin="lower",
        aspect="auto",
        extent=[t_min, t_max, freqs[0], freqs[-1]],
        interpolation="bilinear",  # just to make it look smoother
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Vibration signals")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)
    return Image.open(buf).convert("RGB")

def create_input(df : pd.DataFrame) -> tuple:
    FLOOR_DB = -45.0
    TARGET_HW = (112, 112)

    arr = __create_8_channel(df)

    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)  # (H, W)

        # Ensure float32
    arr = arr.astype(np.float32)

    # If not already in dB, convert. If it *is* already in dB, skip this block.
    # arr = np.maximum(arr, EPS)
    # arr_db = 10.0 * np.log10(arr)

    # Apply dynamic range floor and normalize to [0,1]
    arr = arr - arr.max()
    arr = np.clip(arr, FLOOR_DB, 0.0)
    arr_01 = (arr - FLOOR_DB) / (-FLOOR_DB)

    arr_01 = arr_01.astype(np.float32)

    arr_01 = arr_01 ** 0.9

    arr_01 = np.flipud(arr_01)

    cmap = plt.get_cmap('magma')
    img_rgb = cmap(arr_01)[:, :, :3]

    # Convert to tensor and make 3-channel (grayscale -> RGB-style)
    t = torch.from_numpy(img_rgb)  # (H, W)
    t = t.float().permute(2, 0, 1).unsqueeze(0)  # (3, H, W)

    # Add batch dimension for interpolate
    # t = t.unsqueeze(0)                    # (1, 3, H, W)

    # Resize
    t_resized = F.interpolate(
        t, size=TARGET_HW, mode='bilinear', align_corners=False
    )

    tensorImg = TensorImage(t_resized.squeeze(0))  # (3, H, W)
    png = save_pyplot(arr)

    # Remove batch dimension, wrap as TensorImage
    return tensorImg, png

    # Use the dataframe to create a spectogram and return it

def predict(learner: Learner, img: TensorImage) -> bool:
    # Take the tensorImage as input

    threshold = 0.0138
    print("Calculating Loss...")
    # Check whereever the output for the model is under or above the treshhold
    loss = F.mse_loss(img, learner.predict(img)[0])
    print(loss)

    # Return the result
    return loss >= threshold


def upload(url, tensor: TensorImage, pil_img: Image.Image):

    # --- build .pt ---

    pt_buf = BytesIO()
    torch.save(tensor, pt_buf)
    pt_buf.seek(0)

    # --- build .png in memory ---

    png_buf = BytesIO()
    pil_img.save(png_buf, format='PNG')
    png_buf.seek(0)

    data = {}

    files = {
        "model_input": ("tensor.pt", pt_buf, "application/octet-stream"),
        "file": ("sample.png", png_buf, "image/png"),
    }

    header = {
        "X-Unit-Id": str(1)
    }

    requests.post(url, data=data, files=files, headers=header, timeout=30)
    # If the result from the autoencoder is above the threshold
    # Upload for state detection

def consumer_thread(
        df_queue: "queue.Queue[pd.DataFrame]",
        learner: Learner,
        stop_event: threading.Event,
):
    while not stop_event.is_set():
        try:
            print("Fetching dataframe from the queue...")
            df = df_queue.get(timeout=15)
            print("Creating tensor...")
            tensor_img, png = create_input(df)
            print("Making prediction...")
            pred_result = predict(learner, tensor_img)
            if pred_result:
                upload(URL, tensor_img, png)

            time.sleep(10)

        except queue.Empty:
            print("Queue Empty")

def app():
    df_queue: "queue.Queue[pd.DataFrame]" = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    learner = load_learner('./autoencoder.pkl', cpu=True)

    learner.model.eval()
    learner.model.cpu()

    port = '/dev/ttyACM0'
    baudrate = 230400
    print("starting consumer thread...")
    consumer = threading.Thread(
        target=consumer_thread,
        args=(df_queue, learner, stop_event),
        daemon=True,
        name="consumer",
    )
    print("starting collector thread...")
    collector = threading.Thread(
        target=collect_data,
        args=(df_queue, stop_event),
        daemon=True,
        name="collector",
    )

    consumer.start()
    collector.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()

    consumer.join(timeout=2.0)
    collector.join(timeout=2.0)

if __name__ == '__main__':
    app()
