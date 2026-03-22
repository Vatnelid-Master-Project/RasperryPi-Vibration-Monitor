import queue
import threading
import time
import os
from io import BytesIO
from typing import Callable

from dotenv import load_dotenv
import onnxruntime as ort
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from fastai.vision.all import *
import requests
import serial
from fastapi import APIRouter, Depends, FastAPI
from matplotlib import pyplot as plt
from scipy.signal import ShortTimeFFT, butter, filtfilt
from scipy.signal.windows import hamming
from starlette.websockets import WebSocket, WebSocketDisconnect

load_dotenv()

matplotlib.use('Agg')

URL = os.getenv("URL")
UNIT_ID = os.getenv("UNIT_ID")

def process_chunk(chunk):
    # Example: make 1D numpy array, then filter/spectrogram
    tensor_img, pil = create_input(np.asarray(chunk, dtype=float))
    pred = predict(tensor_img)
    if pred:
        upload(URL, tensor_img, pil)
    # ... your pipeline ...
    # __highpass_filter(x, fs) etc.
    print("Processed chunk", len(np.asarray(chunk, dtype=float)))


def __highpass_filter(x, fs, cutoff=0.5):
    print("HPF x.shape =", x.shape, "x.ndim =", x.ndim, "len(last axis) =", x.shape[-1] if x.ndim else "scalar")
    b, a = butter(4, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, x)

def __create_8_channel(data, sampling_fq = 200, nperseg = 64, dynamic_range_db = 50, band_max_hz = None):
    w = hamming(nperseg)
    sft = ShortTimeFFT(w, hop=int(nperseg*0.25), fs=sampling_fq, scale_to='psd')
    sxx = sft.spectrogram(__highpass_filter(data, sampling_fq))

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

def create_input(data) -> tuple:
    FLOOR_DB = -45.0
    TARGET_HW = (112, 112)

    arr = __create_8_channel(data)
    png = save_pyplot(arr)

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


    # Remove batch dimension, wrap as TensorImage
    return tensorImg, png

    # Use the dataframe to create a spectogram and return it

def predict(img: TensorImage) -> bool:
    # Take the tensorImage as input

    threshold = 0.04
    print("Calculating Loss...")

    session = ort.InferenceSession("autoencoder.onnx")

    img_np = img.cpu().numpy() if hasattr(img, 'cpu') else np.array(img)

    if img_np.ndim == 3:
        img_np = np.expand_dims(img_np, axis=0)

    img_np = img_np.astype(np.float32)

    print(f"Input shape: {img_np.shape}")

    output = session.run(None, {'input': img_np})

    # Check whereever the output for the model is under or above the treshhold
    loss = np.mean((img_np - output[0]) ** 2 )
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
        "X-Unit-Id": str(UNIT_ID)
    }

    requests.post(url, data=data, files=files, headers=header, timeout=30)
    # If the result from the autoencoder is above the threshold
    # Upload for state detection
