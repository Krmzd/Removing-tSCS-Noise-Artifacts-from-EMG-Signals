import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import AttentionUNet1D
from visualization import EMGVisualizer

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "unet_denoiser.pth"
window_size = 400
step_size = 200  # We use 50% overlap to make the signal smoother

def run_inference(csv_path, muscle_col):
    # 1. Load the train model
    model = AttentionUNet1D(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # load and pre-processed data
    df = pd.read_csv(csv_path)
    raw_signal = df[muscle_col].fillna(0).values
    
    # Global mean removal (Zero-Centering)
    centered = raw_signal - np.mean(raw_signal)
    
    # Global Max-Abs Scaling
    scale_factor = np.max(np.abs(centered))
    normalized_signal = centered / scale_factor

    # 3. SLIDING WINDOW RECONSTRUCTION
    # We create a buffer to hold the cleaned signal
    output_buffer = np.zeros_like(normalized_signal)
    count_mask = np.zeros_like(normalized_signal) # To average the overlapping parts

    print(f"Cleaning signal: {muscle_col}...")
    
    with torch.no_grad():
        # Loop through the signal with a sliding window
        for start in range(0, len(normalized_signal) - window_size, step_size):
            end = start + window_size
            
            # Prepare window for U-Net [Batch, Channel, Length]
            window_tensor = torch.tensor(normalized_signal[start:end]).float()
            window_tensor = window_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            # THE MODEL PREDICTS THE CLEAN EMG
            prediction = model(window_tensor)
            
            # Convert back to numpy
            cleaned_window = prediction.cpu().squeeze().numpy()
            
            # Add to buffer and update count for averaging
            output_buffer[start:end] += cleaned_window
            count_mask[start:end] += 1

    # Average the overlapping windows for a smooth transition
    final_cleaned = output_buffer / np.maximum(count_mask, 1)
    
    # 4. SCALE BACK TO ORIGINAL UNITS (mV)
    # This puts the 0.02mV reflex back at its real amplitude
    final_cleaned_mv = final_cleaned * scale_factor

    return raw_signal, final_cleaned_mv


if __name__ == "__main__":
    participants_name = {"JK":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"},
                        "kevyn":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"}}   
    
    file_dir = "D:/University/Cutaneous_reflex_final/U-net CNN/data"
    output_base = "D:/University/Cutaneous_reflex_final/U-net CNN/cleaned_data"
    muscle_names = ["1 L BB", "2 L TB", "3 L AD", "4 L PD", "5 R BB", "", "6 R TB", "7 R AD", "8 L PD"]
    
    # 1. Configuration
    for participant in participants_name:
        
        part_out_dir = os.path.join(output_base, participant)

        os.makedirs(part_out_dir, exist_ok=True)

        condition_name = participants_name[participant]
        
        for condition in condition_name:

            current_file_path = f"{file_dir}/{participant}/{condition}.CSV"

            print(f"Now opening: {current_file_path}")

            # CORRECT initialization of an empty dictionary
            cleaned_only_results = {}

            df_orig = pd.read_csv(current_file_path)
            if "9 trigger" in df_orig.columns:
                cleaned_only_results["9 trigger"] = df_orig["9 trigger"].values

            # 2. Loop through each muscle to Clean, Plot, and Store
            for muscle in muscle_names:
                print(f"\n--- Processing Muscle: {muscle} ---")
                
                    # Run the U-Net cleaning
                raw, cleaned = run_inference(current_file_path, muscle)
                    
                    # Store the result for the final CSV
                cleaned_only_results[muscle] = cleaned

                    # 3. VISUALIZATION (Optional: shows a plot for every muscle processed)
                    # plt.figure(figsize=(12, 6))
                    # plt.subplot(2, 1, 1)
                    # plt.plot(raw, color='deepskyblue', alpha=0.5, label='Raw Noisy Signal')
                    # plt.plot(cleaned, color='black', label='U-Net Cleaned', linewidth=0.7)
                    # plt.title(f"Cleaning Result: {muscle}")
                    # plt.legend()

                    # plt.subplot(2, 1, 2)
                    # plt.plot(raw[-10000:], color='deepskyblue', alpha=0.5, label='Original Tail')
                    # plt.plot(cleaned[-10000:], color='black', label='Cleaned Tail')
                    # plt.title("Validation Check: 10s Clean Tail")
                    # plt.legend()
                    
                    # plt.tight_layout()
                    # plt.show() # Close the plot window to continue to the next muscle

                plotting_tool = EMGVisualizer()
                raw_clean_viz = plotting_tool.plot_inference_check(raw=raw, cleaned=cleaned, muscle_name=muscle)
                


            # 4. SAVE TO CSV
            if cleaned_only_results:
                final_df = pd.DataFrame(cleaned_only_results)
                output_name = os.path.join(part_out_dir, f"{condition}_CLEANED.csv")
                final_df.to_csv(output_name, index=False)
                print(f"Saved: {output_name}")
