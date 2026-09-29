import os
import pandas as pd
from test_denoiser import run_inference  
from visualization import EMGVisualizer
from config import Raw_data_path, Validation_results_path
import torch
from torch.utils.data import DataLoader
from config import Device, Processed_data_path, Model_path
from data_loader import EMGdataset
from model import AttentionUNet1D
from loss import synthetic_metrics

viz_valid = EMGVisualizer(fs=2000)

# Only these will be plotted (use [] for no plots at all)
plot_muscles = ["1 L BB"]
plot_conditions = ["BB_tSCS_before_BLT"]

def validation_result(noisy_file, muscle, make_plots=False):
    """Cleans one muscle of a real tSCS recording and (optionally) plots the result."""

    print("Step 1: Running U-Net Denoising...")

    raw, cleaned = run_inference(noisy_file, muscle)

    if make_plots:
        print("  Generating plots...")
        viz_valid.plot_inference_check(raw=raw, cleaned=cleaned, muscle_name=muscle)
        viz_valid.plot_residual(raw=raw, cleaned=cleaned)
    
    return cleaned

                

def synthetic_evaluation(participants):
    """Part 1: metrics on synthetic .pt data, where the true clean signal is known."""
    model = AttentionUNet1D(n_channels=1, n_classes=1).to(Device)
    model.load_state_dict(torch.load(Model_path, map_location=Device))
    model.eval()

    rows = []
    for p in participants:
        loader = DataLoader(EMGdataset(Processed_data_path, [p]), batch_size=256, shuffle=False)
        noisy_all, clean_all, pred_all = [], [], []
        with torch.no_grad():
            for noisy, clean in loader:
                pred_all.append(model(noisy.to(Device)).cpu())
                noisy_all.append(noisy)
                clean_all.append(clean)

        result = synthetic_metrics(torch.cat(clean_all).numpy().ravel(),
                                   torch.cat(noisy_all).numpy().ravel(),
                                   torch.cat(pred_all).numpy().ravel())
        result["participant"] = p
        rows.append(result)

    df = pd.DataFrame(rows).set_index("participant")
    print(df.round(4).T)
    os.makedirs(Validation_results_path, exist_ok=True)
    df.to_csv(os.path.join(Validation_results_path, "synthetic_metrics.csv"))


if __name__ == "__main__":

    # Part 1: synthetic metrics. NS, MY = validation set, YK = never seen in training
    synthetic_evaluation(["NS", "MY", "YK"])

    participants_name = {"NS":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"},
                        "MY":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"}}   
    
    file_dir = Raw_data_path
    output_dir = Validation_results_path    
    muscle_names = ["1 L BB", "2 L TB", "3 L AD", "4 L PD", "5 R BB", "6 R TB", "7 R AD", "8 R PD"]
    # 1. Configuration
    for participant in participants_name:

        part_out_folder = os.path.join(output_dir, participant)

        os.makedirs(part_out_folder, exist_ok=True)

        condition_name = participants_name[participant]
        
        for condition in condition_name:
            
            noisy_path = f"{file_dir}/{participant}/{condition}.CSV"

            # correct initialization of an empty dictionary
            cleaned_only_results = {}
            # correct initialization of an empty dictionary
            cleaned_only_results = {}

            df_orig = pd.read_csv(noisy_path)
            if "9 trigger" in df_orig.columns:

                cleaned_only_results["9 trigger"] = df_orig["9 trigger"].values

            # Loop through each muscle to Clean, Plot, and Store
            for muscle in muscle_names:
                if muscle in df_orig.columns:
                
                    make_plots = muscle in plot_muscles and condition in plot_conditions
                    cleaned_data = validation_result(noisy_path, muscle, make_plots)
                            
                    cleaned_only_results[muscle] = cleaned_data
          

            # save to csv
            if cleaned_only_results:
                final_df = pd.DataFrame(cleaned_only_results)
                output_name = os.path.join(part_out_folder, f"{condition}_CLEANED.csv")
                final_df.to_csv(output_name, index=False)
                print(f"Saved: {output_name}")
