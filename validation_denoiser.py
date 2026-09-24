import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_denoiser import run_inference  
from process import Process       
from visualization import EMGVisualizer 

viz_valid = EMGVisualizer(fs=2000)

def validation_result(noisy_file, clean_file, muscle):
    
    """
    This function handles the COMPARISON.
    It takes the Noisy file (to clean) and the Clean file (as the Truth).
    """

    print("Step 1: Running U-Net Denoising...")

    raw, cleaned = run_inference(noisy_file , muscle)


    df_truth = pd.read_csv(clean_file)
    truth_data = df_truth[muscle].fillna(0).values
    truth_data = truth_data - np.mean(truth_data) 

    # Align lengths because recordings have different sizes
    min_len = min(len(cleaned), len(truth_data))
    y_pred = cleaned[:min_len]
    y_true = truth_data[:min_len]

    print("  Generating Residual and Distribution plots...")
    # plot
    viz_valid.plot_inference_check(raw=raw, cleaned=cleaned, muscle_name=muscle)
    viz_valid.plot_residual(target_sig=y_true, pred_sig=y_pred)
    viz_valid.plot_error_distribution(target=y_true, pred=y_pred)
    
    return cleaned

                

if __name__ == "__main__":

    participants_name = {"NS":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"},
                        "MY":{"BB_tSCS_before_BLT", "TB_tSCS_before_BLT", "AD_PD_tSCS_before_BLT", "BB_tSCS_after_BLT", "TB_tSCS_after_BLT", "AD_PD_tSCS_after_BLT"}}   
    
    file_dir = Raw_data_path
    output_dir = Validation_results_path    
    muscle_names = ["1 L BB", "2 L TB", "3 L AD", "4 L PD", "5 R BB", "", "6 R TB", "7 R AD", "8 L PD"]
    # 1. Configuration
    for participant in participants_name:

        part_out_folder = os.path.join(output_dir, participant)

        os.makedirs(part_out_folder, exist_ok=True)

        condition_name = participants_name[participant]
        
        for condition in condition_name:

            noisy_path = f"{file_dir}/{participant}/{condition}.CSV"
            # find the corresponding clean file (no tSCS)
            clean_path = noisy_path.replace("tSCS", "no_tSCS")

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                print(f"\nProcessing: {participant} | {condition}")

            # correct initialization of an empty dictionary
            cleaned_only_results = {}

            df_orig = pd.read_csv(noisy_path)
            if "9 trigger" in df_orig.columns:

                cleaned_only_results["9 trigger"] = df_orig["9 trigger"].values

            # Loop through each muscle to Clean, Plot, and Store
            for muscle in muscle_names:
                if muscle in df_orig.columns:
                
                    cleaned_data = validation_result(noisy_path, clean_path, muscle)
                            
                    cleaned_only_results[muscle] = cleaned_data
          

            # save to csv
            if cleaned_only_results:
                final_df = pd.DataFrame(cleaned_only_results)
                output_name = os.path.join(part_out_folder, f"{condition}_CLEANED.csv")
                final_df.to_csv(output_name, index=False)
                print(f"Saved: {output_name}")
