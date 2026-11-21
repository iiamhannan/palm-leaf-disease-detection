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


