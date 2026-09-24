import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Folder that contains this file (the Attention-U-Net project)
Base_dir = Path(__file__).resolve().parent
# Folder above it ("U-net CNN"), where data/ and processed_data/ live
Root_dir = Base_dir.parent

# Paths: set them in .env, otherwise these defaults are used
Raw_data_path = Path(os.getenv("RAW_DATA_PATH", Root_dir / "data"))
Processed_data_path = Path(os.getenv("PROCESSED_DATA_PATH", Root_dir / "processed_data"))
Cleaned_data_path = Path(os.getenv("CLEANED_DATA_PATH", Root_dir / "cleaned_data"))
Validation_results_path = Path(os.getenv("VALIDATION_RESULTS_PATH", Root_dir / "validation_results"))
Model_path = Path(os.getenv("MODEL_PATH", Base_dir / "unet_denoiser.pth"))


# Device: NVIDIA GPU (Windows) -> Apple GPU (Mac) -> CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


Device = get_device()
