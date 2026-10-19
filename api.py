from fastapi import FastAPI

app = FastAPI(title="EMG Denoiser API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": "Attention U-Net 1D",
        "task": "EMG denoising",
        "window_size": 400,
    }
