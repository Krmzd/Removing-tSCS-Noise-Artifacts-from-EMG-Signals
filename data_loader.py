import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from process import Process

class EMGdataset(Dataset):
    def __init__(self, processed_path, participants):

        self.all_noisy = []
        self.all_clean = []

        print(f"Loading data for participants: {participants}")

        for p in participants:
            part_folder = os.path.join(processed_path, p) # building the address string

            if not os.path.exists(part_folder):
                raise ValueError(f"Participant folder not found: {part_folder}")

            pt_files = [
                f for f in os.listdir(part_folder)
                if f.endswith(".pt")
            ]

            if not pt_files:
                raise ValueError(f"No .pt files inside {part_folder}")

            for f in pt_files:
                path = os.path.join(part_folder, f)

                n_batch, c_batch = torch.load(path)

                if n_batch.shape != c_batch.shape:
                    raise ValueError(f"Shape mismatch in {f}")


                self.all_noisy.append(n_batch)
                self.all_clean.append(c_batch)

        self.all_noisy = torch.cat(self.all_noisy, dim=0)
        self.all_clean = torch.cat(self.all_clean, dim=0)

        print("Final dataset shape:", self.all_noisy.shape)

    def __len__(self):
        return self.all_noisy.shape[0]
    
    def __getitem__ (self, index):
        return self.all_noisy[index], self.all_clean[index]
    
def extract_pure_noise(signal, peak_indices, pre_width=10, post_width=35):

    interpolated = signal.copy()
    for p in peak_indices:
        start = max(0, p - pre_width)
        end = min(len(signal) - 1, p + post_width)
        # Bridge over the spike
        val_start = signal[start]
        val_end = signal[end]
        interpolated[start:end] = np.linspace(val_start, val_end, num=end-start)
    
    # Pure noise = Original minus the bridges
    return signal - interpolated


def pre_process_batched(raw_path, output_path, window_size=400):

    process_data = Process()
    column = {
        "L": {"AD": "3 L AD",
                "PD": "4 L PD",
                "BB": "1 L BB",
                "TB": "2 L TB",
        },
            "R": {"AD": "7 R AD",
                "PD": "8 R PD",
                "BB": "5 R BB",
                "TB": "6 R TB",
        }
    }

    conditions = [
        ("AD_PD_tSCS_before_BLT", "AD_PD_no_tSCS_before_BLT"),
        ("AD_PD_tSCS_after_BLT",  "AD_PD_no_tSCS_after_BLT"),
        ("BB_tSCS_before_BLT",   "BB_no_tSCS_before_BLT"),
        ("BB_tSCS_after_BLT",    "BB_no_tSCS_after_BLT"),
        ("TB_tSCS_before_BLT",  "TB_no_tSCS_before_BLT"),
        ("TB_tSCS_after_BLT",   "TB_no_tSCS_after_BLT")
    ]

    participants = [
    p for p in os.listdir(raw_path)
    if os.path.isdir(os.path.join(raw_path, p))
    ]
    
    for part in tqdm(participants, desc="Participants", leave=True, position=0):
        part_folder = os.path.join(raw_path, part)
        
        participant_output = os.path.join(output_path, part)
        os.makedirs(participant_output, exist_ok=True)

        for side in ["L", "R"]:
            for muscle, col in column[side].items():
                if muscle in ["BB", "TB"]:
                    h_mult = 3.0
                    pre_w, post_w = 10, 25
                else:
                    h_mult = 8.0
                    pre_w, post_w = 15, 40

                for noisy_name, clean_name in conditions:
                    if muscle in noisy_name:
                        n_csv = os.path.join(part_folder, f"{noisy_name}.csv")   
                        c_csv = os.path.join(part_folder, f"{clean_name}.csv") 

                        if os.path.exists(n_csv) and os.path.exists(c_csv):

                            df_n = pd.read_csv(n_csv, usecols=[col]).fillna(0)
                            df_c = pd.read_csv(c_csv, usecols=[col]).fillna(0)
                            
                            # Global Mean Removal
                            df_n = process_data.offset_removal(df_n)
                            df_c = process_data.offset_removal(df_c)

                            # Max-Abs Scaling (Numerical Range [-1, 1])
                            max_value = df_n.abs().max().iloc[0] 

                            n_scaled = df_n[col] / max_value
                            c_scaled = df_c[col] / max_value

                            tail_std = np.std(n_scaled[-20000:])
                            thresh = tail_std * h_mult
                            
                            # noise extraction
                            # Find peaks on the noisy signal
                            peaks, _ = find_peaks(np.abs(n_scaled), height=thresh, distance=40)
                        
                            pure_noise = np.zeros_like(n_scaled)
                            for p in peaks:
                                s, e = max(0, p - pre_w), min(len(n_scaled), p + post_w)
                                # Bridge to isolate spike shape
                                v1, v2 = n_scaled[s], n_scaled[e-1]
                                bridge = np.linspace(v1, v2, num=e-s)
                                # Paste spike minus bridge into the zeroed-out array
                                pure_noise[s:e] = n_scaled[s:e] - bridge

                            # Synthetic Mixing
                            min_len = min(len(c_scaled), len(pure_noise))
                            train_target = c_scaled.values[:min_len]
                            # Input = Clean Signal + Noise Sticker
                            train_input = train_target + pure_noise[:min_len]

                            num_w = min_len // window_size
                            
                            n_list, c_list = [], []

                            for i in range(num_w):
                                start = i * window_size
                            
                                # Extract slice and add channel dimension
                                n_win = torch.tensor(train_input[start:start+window_size]).float().unsqueeze(0)
                                c_win = torch.tensor(train_target[start:start+window_size]).float().unsqueeze(0)

                                n_list.append(n_win)
                                c_list.append(c_win)

                            if n_list:
                                suffix = "_".join(noisy_name.split("_")[-2:]) 
                                file_name = f"{side}_{muscle}_{suffix}.pt"
                                torch.save((torch.stack(n_list), torch.stack(c_list)), 
                                        os.path.join(participant_output, file_name))
                                   
                                
def get_loader(processed_path, train_participants, val_participants, batch_size=32):
    train_set = EMGdataset(processed_path, train_participants)
    val_set = EMGdataset(processed_path, val_participants)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
# if __name__ == "__main__":
#     raw = "D:/University/Cutaneous_reflex_final/U-net CNN/data"
#     processed = "D:/University/Cutaneous_reflex_final/U-net CNN/processed_data"
#     pre_process_batched(raw, processed)
 
# def check_pt_file(file_path):
#     # 1. Load the data
#     # Your code saved them as: torch.save((torch.stack(n_list), torch.stack(c_list)), ...)
#     noisy_windows, clean_windows = torch.load(file_path)

#     # Print the Shapes
#     # Expected: [Number of Windows, 1, 400]
#     print(f"--- Metadata for: {file_path} ---")
#     print(f"Noisy Tensor Shape: {noisy_windows.shape}")
#     print(f"Clean Tensor Shape: {clean_windows.shape}")

#     # Print the Value Ranges
#     # This ensures your normalization worked
#     print(f"Noisy Max: {noisy_windows.max().item():.4f} | Noisy Min: {noisy_windows.min().item():.4f}")
#     print(f"Clean Max: {clean_windows.max().item():.4f} | Clean Min: {clean_windows.min().item():.4f}")

#     # Visual Check: Plot one specific window (e.g., window #10)
#     window_idx = 10
#     plt.figure(figsize=(12, 5))
    
#     # We use [window_idx, 0, :] to get the 400 samples
#     input_data = noisy_windows[window_idx, 0, :].numpy()
#     target_data = clean_windows[window_idx, 0, :].numpy()

#     plt.plot(input_data, label='Input (Synthetic Noisy)', color='crimson', alpha=0.8)
#     plt.plot(target_data, label='Target (Original Clean)', color='steelblue', alpha=0.8)
    
#     plt.title(f"Verification of Window {window_idx}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
