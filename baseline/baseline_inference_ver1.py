#!/usr/bin/env python
# nrse_baseline_inference.py

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import librosa
import time
from transformers import AutoModel
import pickle

# Set up path to include the NRSE_baseline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NRSE_baseline modules
import net
from src.data.emotion_dataset import EmotionDataset
# import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with NRSE baseline pretrained model")
    parser.add_argument("--ssl_type", type=str, default="wavlm-large", 
                        help="Type of SSL model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to directory containing model weights (final_ssl.pt, final_pool.pt, final_ser.pt)")
    parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling",
                        help="Type of pooling (must match the pretrained model)")
    parser.add_argument("--head_dim", type=int, default=1024,
                        help="Dimension of head layer")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to directory of audio files for inference")
    parser.add_argument("--audio_list", type=str, default=None,
                        help="Path to file that has list of audio files for inference, if not specified scan all audion in audio_dir")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save inference results")
    parser.add_argument("--task", type=str, default="dimensional", choices=["categorical", "dimensional"],
                        help="Type of emotion recognition task")
    return parser.parse_args()


def load_audio_file(file_path, sample_rate=16000):
    """Load and preprocess a single audio file without caching."""
    try:
        # Load audio with librosa with caching disabled
        waveform, sr = librosa.load(file_path, sr=sample_rate)
        
        # Convert to tensor
        waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
        
        # Create mask (all ones since we're using the entire audio)
        mask = torch.ones_like(waveform)
        
        return waveform, mask
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None


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


def normalize_waveform(waveform, wav_mean, wav_std):
    """Normalize waveform using mean and std."""
    return (waveform - wav_mean) / wav_std


def process_audio_file(file_path, ssl_model, pool_model, ser_model, wav_mean, wav_std, task):
    """Process a single audio file and return emotion predictions."""
    # Load audio
    waveform, mask = load_audio_file(file_path)
    if waveform is None:
        return None
    
    # Normalize waveform
    waveform = normalize_waveform(waveform, wav_mean, wav_std)
    
    # Move to GPU
    waveform = waveform.cuda()
    mask = mask.cuda()
    
    # Inference
    with torch.no_grad():
        # Extract SSL features
        ssl_features = ssl_model(waveform, attention_mask=mask).last_hidden_state
        
        # Apply pooling
        pooled_features = pool_model(ssl_features, mask)
        
        # Get emotion predictions
        emotion_preds = ser_model(pooled_features)
    
    # Return predictions based on task type
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path)
    }
    
    if task == "dimensional":
        # For dimensional, return arousal, valence, dominance
        result.update({
            'arousal': emotion_preds[0, 0].item(),
            'valence': emotion_preds[0, 1].item(),
            'dominance': emotion_preds[0, 2].item()
        })
    else:
        # For categorical, return emotion class and probabilities
        probs = torch.softmax(emotion_preds, dim=1)[0]
        emotion_idx = torch.argmax(emotion_preds, dim=1).item()
        emotion_map = {v: k for k, v in EmotionDataset.EMOTIONS_MAP.items()}
        
        result.update({
            'emotion': emotion_map.get(emotion_idx, f"Unknown-{emotion_idx}"),
            'probabilities': probs.cpu().numpy().tolist()
        })
    
    return result


def process_audio_directory(directory, ssl_model, pool_model, ser_model, wav_mean, wav_std, task):
    """Process all audio files in a directory."""
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files in {directory}")
    
    # Process each file
    results = []
    for file_path in tqdm(audio_files):
        result = process_audio_file(file_path, ssl_model, pool_model, ser_model, wav_mean, wav_std, task)
        if result:
            results.append(result)
    
    return results


def save_results(results, output_dir, task, create_detailed_results):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    import csv
    
    if task == "dimensional":
        csv_path = os.path.join(output_dir, "dimensional_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'arousal', 'valence', 'dominance'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'file_name': result['file_name'],
                    'A': result['arousal'],
                    'V': result['valence'],
                    'D': result['dominance']
                })
    else:
        csv_path = os.path.join(output_dir, "categorical_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'emotion'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'file_name': result['file_name'],
                    'C': result['emotion']
                })
    
    # Save detailed results as JSON
    if create_detailed_results:
      import json
      json_path = os.path.join(output_dir, "detailed_results.json")
      with open(json_path, 'w') as f:
          json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Validate input
    if args.audio_file is None and args.audio_dir is None:
        print("Error: Must specify either --audio_file or --audio_dir")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    ssl_model, pool_model, ser_model, wav_mean, wav_std = load_model(args)
    
    # Process audio
    if args.audio_file:
        # Single file
        result = process_audio_file(args.audio_file, ssl_model, pool_model, ser_model, wav_mean, wav_std, args.task)
        results = [result] if result else []
    else:
        # Directory
        results = process_audio_directory(args.audio_dir, ssl_model, pool_model, ser_model, wav_mean, wav_std, args.task)
    
    # Save results
    if results:
        save_results(results, args.output_dir, args.task, False)
        print(f"Processed {len(results)} files")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()