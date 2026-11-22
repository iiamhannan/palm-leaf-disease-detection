import base64
import io
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE = Path(__file__).parent
CLASSES = ["brown spots", "healthy", "white scale"]  # same order as training
NAMES = {"brown spots": "Brown Spot", "healthy": "Healthy", "white scale": "White Scale"}

sess = ort.InferenceSession(str(BASE / "palm_model_v6.onnx"), providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Weather rules (edit to taste)
RAIN_PROB = 60   # percent, highest chance in the next 24 h
RAIN_MM = 2.0    # mm, total in the next 24 h
WIND_MAX = 20    # km/h, above this spray drifts
OIL_MAX_TEMP = 35  # deg C, mineral oil can scorch leaves above this


def prep(img):
    img = img.resize((224, 224), Image.BILINEAR)  # v6 is trained on sharp input (no 112 downsample)
    a = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def heatmap(img, cam):
    """Overlay a class-activation heatmap on the photo; returns a data URL."""
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = img.copy()
    base.thumbnail((512, 512))
    c = Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    v = np.asarray(c, dtype=np.float32) / 255
    jet = np.stack([np.clip(1.5 - np.abs(4 * v - 3), 0, 1),
                    np.clip(1.5 - np.abs(4 * v - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * v - 1), 0, 1)], -1)
    out = 0.55 * np.asarray(base, dtype=np.float32) / 255 + 0.45 * jet
    buf = io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8)).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def get_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "hourly": "precipitation_probability,precipitation",
    }
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
        d = r.json()
    h, cur = d["hourly"], d["current"]
    now = cur["time"][:13] + ":00"
    i = next((k for k, t in enumerate(h["time"]) if t >= now), 0)
    prob = max(x or 0 for x in h["precipitation_probability"][i:i + 24])
    mm = sum(x or 0 for x in h["precipitation"][i:i + 24])
    return {
        "temperature": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"],
        "wind": cur["wind_speed_10m"], "rain_probability": prob, "rain_mm": round(mm, 1),
        "rain_expected": prob >= RAIN_PROB or mm >= RAIN_MM,
    }