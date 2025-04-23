#!/usr/bin/env python
# baseline_inference_simple.py

import os
import sys
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import argparse
from transformers import AutoModelForAudioClassification
from transformers import AutoModel

# Set up path to include the NRSE_baseline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NRSE_baseline modules
import net
from src.data.emotion_dataset import EmotionDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Audio emotion prediction using pretrained model")
    parser.add_argument("--model_name", type=str, default="3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes",
                        help="Name of the pretrained model")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to directory containing audio files")
    parser.add_argument("--audio_list", type=str, default=None,
                        help="Path to file that has list of audio files and ground truth")
    parser.add_argument("--output_file", type=str, default="emotion_predictions.csv",
                        help="Path to output CSV file")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU for inference if available")
    return parser.parse_args()

def load_audio_list(list_path, audio_dir):
    """Load list of audio files to process from a text file."""
    print(f"Loading audio list from {list_path}")
    audio_files = []
    labels = []
    
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: Audios/MSP-PODCAST_0408_0129.wav; H; A:5.4; V:4.0; D:5.6;
            parts = line.split(';')
            if len(parts) < 2:
                print(f"Skipping invalid line: {line}")
                continue
                
            # Get file path
            file_path = parts[0].strip()
            # Make path absolute if it's relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(audio_dir, file_path)
            
            # Extract label information
            label_info = {}
            
            # Extract emotion class label
            if len(parts) > 1:
                label_info['emotion'] = parts[1].strip()
            
            # Extract dimensional values if present
            for i in range(2, len(parts)):
                part = parts[i].strip()
                if not part:
                    continue
                    
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        value = float(value)
                        label_info[key] = value
                    except ValueError:
                        label_info[key] = value
            
            audio_files.append(file_path)
            labels.append(label_info)
    
    print(f"Loaded {len(audio_files)} files from list")
    return audio_files, labels

def get_audio_files_from_directory(directory):
    """Get all audio files from a directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files in {directory}")
    return audio_files


def process_audio_files(files_list, model, mean, std, device, gt_labels=None):
    """Process audio files following exactly the example pattern."""
    # Get label mapping directly from the model's config
    label_map = model.config.id2label
    
    # Create mapping from index to emotion code
    idx_to_emotion = {}
    for idx, label in label_map.items():
        # Convert from full emotion name to single letter code
        if label == "Angry":
            idx_to_emotion[int(idx)] = "A"
        elif label == "Sad":
            idx_to_emotion[int(idx)] = "S"
        elif label == "Happy":
            idx_to_emotion[int(idx)] = "H"
        elif label == "Surprise":
            idx_to_emotion[int(idx)] = "U"
        elif label == "Fear":
            idx_to_emotion[int(idx)] = "F"
        elif label == "Disgust":
            idx_to_emotion[int(idx)] = "D"
        elif label == "Contempt":
            idx_to_emotion[int(idx)] = "C"
        elif label == "Neutral":
            idx_to_emotion[int(idx)] = "N"
        else:
            idx_to_emotion[int(idx)] = label
    
    results = []
    
    for i, file_path in enumerate(tqdm(files_list, desc="Processing files")):
        try:
            # Load and normalize audio exactly as in the example
            raw_wav, _ = librosa.load(file_path, sr=model.config.sampling_rate)
            norm_wav = (raw_wav - mean) / (std + 0.000001)
            
            # Create mask and prepare wavs exactly as in the example
            mask = torch.ones(1, len(norm_wav)).to(device)
            wavs = torch.tensor(norm_wav).unsqueeze(0).to(device)
            
            # Predict using the model
            with torch.no_grad():
                pred = model(wavs, mask)
            
            # Process predictions exactly as in the example
            logits = pred.logits if hasattr(pred, 'logits') else pred
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Get predicted class
            pred_idx = torch.argmax(logits, dim=1).item()
            predicted_emotion = idx_to_emotion.get(pred_idx, f"Unknown-{pred_idx}")
            
            # Create result
            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'predicted_emotion': predicted_emotion,
                'confidence': probs[pred_idx].item()
            }
            
            # Add ground truth if available
            if gt_labels and i < len(gt_labels):
                for key, value in gt_labels[i].items():
                    result[f'gt_{key}'] = value
            
            # Add all probabilities
            for idx, emotion in idx_to_emotion.items():
                if idx < len(probs):
                    result[f'{emotion}_prob'] = probs[idx].item()
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        
        # Clear GPU cache periodically
        if i % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results

def load_model(args):
    """Load the pretrained model components."""
    print(f"Loading pretrained model from {args.model_path}")
    
    # Load SSL model
    ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")
    ssl_model.freeze_feature_encoder()
    ssl_model.load_state_dict(torch.load(os.path.join(args.model_path, "final_ssl.pt")))
    ssl_model.eval()
    ssl_model.cuda()
    
    # Get feature dimension
    feat_dim = ssl_model.config.hidden_size
    
    # Load pooling model
    pool_net = getattr(net, args.pooling_type)
    attention_pool_type_list = ["AttentiveStatisticsPooling"]
    if args.pooling_type in attention_pool_type_list:
        pool_model = pool_net(feat_dim)
        pool_model.load_state_dict(torch.load(os.path.join(args.model_path, "final_pool.pt")))
    else:
        pool_model = pool_net()
    
    pool_model.eval()
    pool_model.cuda()
    
    # Load SER model
    concat_pool_type_list = ["AttentiveStatisticsPooling"]
    dh_input_dim = feat_dim * 2 if args.pooling_type in concat_pool_type_list else feat_dim
    
    if args.task == "dimensional":
        ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 3, dropout=0.0)
    else:  # categorical
        ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 8, dropout=0.0)
    
    ser_model.load_state_dict(torch.load(os.path.join(args.model_path, "final_ser.pt")))
    ser_model.eval()
    ser_model.cuda()
    
    # Load normalization stats
    norm_stat_path = os.path.join(args.model_path, "train_norm_stat.pkl")
    if os.path.exists(norm_stat_path):
        with open(norm_stat_path, 'rb') as f:
            norm_stats = pickle.load(f)
        wav_mean = norm_stats[0]
        wav_std = norm_stats[1]
    else:
        wav_mean = 0.0
        wav_std = 1.0
        print("Warning: No normalization stats found, using default values")
    
    return ssl_model, pool_model, ser_model, wav_mean, wav_std

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model: {args.model_name}")
    audio_model = AutoModelForAudioClassification.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )
    # Move model to device
    audio_model = audio_model.to(device)
    audio_model.eval()
    print(audio_model.config.id2label)
    
    # Get mean and std from model config
    mean = audio_model.config.mean
    std = audio_model.config.std
    
    # Get audio files and ground truth (if available)
    files_list = []
    gt_labels = None
    
    if args.audio_list:
        files_list, gt_labels = load_audio_list(args.audio_list, args.audio_dir)
    else:
        files_list = get_audio_files_from_directory(args.audio_dir)
    
    # Process audio files
    results = process_audio_files(files_list, audio_model, mean, std, device, gt_labels)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")
    
    # Print summary statistics if ground truth is available
    if 'gt_emotion' in df.columns:
        correct = (df['predicted_emotion'] == df['gt_emotion']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        confusion = pd.crosstab(df['gt_emotion'], df['predicted_emotion'], 
                              rownames=['Ground Truth'], colnames=['Predicted'])
        print(confusion)

if __name__ == "__main__":
    main()