#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion recognition predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV file")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth labels CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--split", type=str, default=None,
                        help="Split set to evaluate (e.g., 'Train', 'Development', 'Test1'). If not specified, all samples are evaluated.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions and ground truth
    print(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
    
    print(f"Loading ground truth from {args.ground_truth}")
    ground_truth_df = pd.read_csv(args.ground_truth)
    
    # Filter ground truth by split set if specified
    if args.split:
        if 'Split_Set' in ground_truth_df.columns:
            original_size = len(ground_truth_df)
            ground_truth_df = ground_truth_df[ground_truth_df['Split_Set'] == args.split]
            filtered_size = len(ground_truth_df)
            print(f"Filtered ground truth to '{args.split}' split: {filtered_size} samples (from {original_size})")
            
            if filtered_size == 0:
                print(f"Error: No samples found in split '{args.split}'. Available splits: {ground_truth_df['Split_Set'].unique()}")
                return
        else:
            print("Warning: 'Split_Set' column not found in ground truth. Cannot filter by split.")
    
    # Map filenames in predictions to remove path and potentially SNR info
    def clean_filename(filename):
        # Extract base filename without path and extension
        base = os.path.basename(filename)
        # If the filename contains SNR info (e.g., _snr4_), remove it
        if "_snr" in base:
            parts = base.split("_snr")
            base = parts[0] + ".wav"  # Add .wav extension back
        return base
    
    # Check if 'file_name' or 'Filename' is used in predictions
    filename_column = None
    for possible_col in ['file_name', 'Filename', 'FileName', 'filename']:
        if possible_col in predictions_df.columns:
            filename_column = possible_col
            break
            
    if filename_column is None:
        print("Error: Could not find filename column in predictions. Please check column names.")
        return
        
    predictions_df['CleanFileName'] = predictions_df[filename_column].apply(clean_filename)
    
    # Merge dataframes on filenames
    print("Merging predictions with ground truth")
    merged_df = pd.merge(
        predictions_df, 
        ground_truth_df, 
        left_on='CleanFileName', 
        right_on='FileName', 
        how='inner'
    )
    
    print(f"Found {len(merged_df)} matched samples out of {len(predictions_df)} predictions")
    
    if len(merged_df) == 0:
        print("No matching samples found. Please check filenames in both datasets.")
        return
    
    # Define valid emotion labels
    valid_emotions = {'A', 'H', 'S', 'U', 'F', 'D', 'C', 'N'}
    emotion_labels = {
        'A': 'Anger',
        'H': 'Happiness',
        'S': 'Sadness',
        'F': 'Fear',
        'U': 'Surprise',
        'D': 'Disgust',
        'C': 'Contempt',
        'N': 'Neutral'
    }
    
    # Filter out rows with X or O labels
    print(f"Total samples before filtering: {len(merged_df)}")
    merged_df = merged_df[merged_df['EmoClass'].isin(valid_emotions)]
    print(f"Total samples after filtering out X and O: {len(merged_df)}")
    
    # Check if we have enough samples after filtering
    if len(merged_df) == 0:
        print("No valid emotion labels found after filtering out X and O.")
        return
    
    # Check which column contains predictions
    pred_column = None
    possible_columns = ['predicted_emotion', 'PredictedEmotion', 'Prediction', 'prediction']
    for col in possible_columns:
        if col in merged_df.columns:
            pred_column = col
            break
    
    if pred_column is None:
        # Try to detect prediction column automatically
        for col in merged_df.columns:
            if 'pred' in col.lower():
                pred_column = col
                break
    
    if pred_column is None:
        print("Could not find prediction column. Please specify the column name containing predictions.")
        return
    
    # Prepare data for metrics calculation
    y_true = merged_df['EmoClass'].values
    y_pred = merged_df[pred_column].values
    
    # Filter out any predictions that are not in the valid emotions set
    valid_indices = [i for i, label in enumerate(y_pred) if label in valid_emotions]
    if len(valid_indices) < len(y_pred):
        print(f"Filtered out {len(y_pred) - len(valid_indices)} predictions that were not valid emotion labels.")
        y_true = np.array([y_true[i] for i in valid_indices])
        y_pred = np.array([y_pred[i] for i in valid_indices])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert report to DataFrame for easier handling
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    filename_prefix = f"{args.split}_" if args.split else ""
    report_path = os.path.join(args.output_dir, f"{filename_prefix}classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # Compute and plot confusion matrix
    labels = sorted(list(valid_emotions))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[emotion_labels[k] for k in labels],
               yticklabels=[emotion_labels[k] for k in labels])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title = f"Confusion Matrix - {args.split}" if args.split else "Confusion Matrix"
    plt.title(title)
    
    cm_path = os.path.join(args.output_dir, f"{filename_prefix}confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save per-class F1 scores
    class_f1 = {}
    for emotion_key in emotion_labels.keys():
        if emotion_key in report:
            class_f1[emotion_labels[emotion_key]] = report[emotion_key]['f1-score']
    
    # Plot per-class F1 scores
    plt.figure(figsize=(10, 6))
    classes = list(class_f1.keys())
    f1_values = list(class_f1.values())
    
    plt.bar(classes, f1_values)
    plt.axhline(y=macro_f1, color='r', linestyle='-', label=f'Macro F1: {macro_f1:.4f}')
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score')
    title = f"F1 Score by Emotion Class - {args.split}" if args.split else "F1 Score by Emotion Class"
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    
    f1_path = os.path.join(args.output_dir, f"{filename_prefix}f1_by_class.png")
    plt.tight_layout()
    plt.savefig(f1_path)
    plt.close()
    print(f"F1 score plot saved to {f1_path}")
    
    # Calculate per-class metrics
    class_metrics = {}
    for cls in valid_emotions:
        if cls in report:
            class_metrics[emotion_labels[cls]] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': report[cls]['support']
            }
    
    # Save summary metrics
    split_info = f"Split: {args.split}" if args.split else "Split: All"
    summary = {
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1,
        'Number of samples': len(y_true),
        'Split': args.split if args.split else "All"
    }
    
    # Add per-class F1 scores to summary
    for cls, metrics in class_metrics.items():
        summary[f"{cls} F1"] = metrics['f1-score']
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, f"{filename_prefix}summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")

    # Print class distribution in dataset
    class_distribution = {emotion_labels[k]: sum(y_true == k) for k in valid_emotions if k in y_true}
    print("\nClass distribution in evaluation set:")
    for emotion, count in class_distribution.items():
        percentage = (count / len(y_true)) * 100
        print(f"{emotion}: {count} samples ({percentage:.2f}%)")
    
    # Save class distribution to CSV
    dist_df = pd.DataFrame([{
        'Emotion': emotion,
        'Count': count,
        'Percentage': (count / len(y_true)) * 100
    } for emotion, count in class_distribution.items()])
    
    dist_path = os.path.join(args.output_dir, f"{filename_prefix}class_distribution.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"Class distribution saved to {dist_path}")

if __name__ == "__main__":
    main()