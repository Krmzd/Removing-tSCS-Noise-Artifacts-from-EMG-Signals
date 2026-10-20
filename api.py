from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import WINDOW_SIZE, denoise_signal, load_model

app = FastAPI(title="EMG Denoiser API")

# Load the trained model once, when the API starts
model = load_model()


class SignalIn(BaseModel):
    signal: List[float]


class SignalOut(BaseModel):
    cleaned: List[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"model": "Attention U-Net 1D", "task": "EMG denoising", "window_size": WINDOW_SIZE}


@app.post("/denoise", response_model=SignalOut)
def denoise(data: SignalIn):
    if len(data.signal) < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Signal must have at least {WINDOW_SIZE} samples.")
    cleaned = denoise_signal(model, data.signal)
    return {"cleaned": cleaned.tolist()}
