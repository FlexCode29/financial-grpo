import os
import json
import pandas as pd
import argparse

def main(args):
    all_metrics = []
    
    print(f"--- Aggregating Cross-Validation Results from: {args.base_dir} ---")
    
    # Find all evaluation result files from the subdirectories
    for i in range(args.k_folds):
        fold_dir = os.path.join(args.base_dir, f"fold_{i}")
        metric_file = os.path.join(fold_dir, "evaluation_metrics.json")
        
        if os.path.exists(metric_file):
            print(f"Found results for fold {i}")
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        else:
            print(f"WARNING: Could not find results for fold {i} at {metric_file}")
            
    if not all_metrics:
        print("No metric files found. Exiting.")
        return

    # Convert the list of metric dictionaries to a DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    # Calculate the mean and standard deviation for each metric
    summary_mean = results_df.mean().rename("Mean")
    summary_std = results_df.std().rename("Std Dev")
    
    final_summary = pd.concat([summary_mean, summary_std], axis=1)
    
    print("\n--- Cross-Validation Final Summary ---")
    print(final_summary)
    print("--------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CV results.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the fold outputs.")
    parser.add_argument("--k_folds", type=int, required=True, help="The total number of folds that were run.")
    args = parser.parse_args()
    main(args)