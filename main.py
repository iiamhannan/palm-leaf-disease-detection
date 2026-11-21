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