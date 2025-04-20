#!/usr/bin/env python
# baseline_inference.py

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import librosa
import pickle
import json
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with NRSE baseline model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to directory containing model weights (should have final_ssl.pt, final_pool.pt, final_ser.pt)")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Path to single audio file for inference")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Path to directory of audio files for inference")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save inference results")
    parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling",
                        help="Type of pooling used in the model")
    parser.add_argument("--head_dim", type=int, default=1024,
                        help="Dimension of head layer")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--skip_ssl", action="store_true",
                        help="Skip loading the SSL model (use if you're having issues)")
    return parser.parse_args()

def load_audio_file(file_path, sample_rate=16000):
    """Load and preprocess a single audio file."""
    try:
        # Load audio with librosa
        waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        
        # Convert to tensor
        waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
        
        # Create mask (all ones since we're using the entire audio)
        mask = torch.ones_like(waveform)
        
        return waveform, mask
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def detect_model_type(model_dir):
    """Detect whether the model is for categorical or dimensional emotion."""
    try:
        # Load the model state dict to check its structure
        state_dict = torch.load(os.path.join(model_dir, "final_ser.pt"))
        
        # Check the shape of the output layer
        if 'out.0.weight' in state_dict:
            output_shape = state_dict['out.0.weight'].shape
            print(f"Detected output shape: {output_shape}")
            
            # If output dimension is 8, it's categorical. If 3, it's dimensional.
            if output_shape[0] == 8:
                print("Detected a categorical emotion model")
                return "categorical"
            elif output_shape[0] == 3:
                print("Detected a dimensional emotion model")
                return "dimensional"
        
        # If we can't determine from the output layer, check if we can infer from directory name
        if 'cat' in model_dir.lower():
            print("Directory name suggests a categorical emotion model")
            return "categorical"
        elif 'dim' in model_dir.lower():
            print("Directory name suggests a dimensional emotion model")
            return "dimensional"
            
        # Default to categorical if we can't determine
        print("Could not determine model type, defaulting to categorical")
        return "categorical"
        
    except Exception as e:
        print(f"Error detecting model type: {e}")
        print("Defaulting to categorical model")
        return "categorical"

def load_model(args):
    """Load the pretrained model components."""
    print(f"Loading pretrained model from {args.model_dir}")
    
    # Import required modules
    sys.path.append(os.getcwd())
    try:
        import net
    except ImportError:
        print("Error importing net module. Make sure NRSE_baseline modules are in your PYTHONPATH")
        sys.exit(1)
    
    # Check if we've set the right environment variable
    if os.environ.get('PYTHONPATH') is None:
        print("PYTHONPATH is not set. Set it to include the NRSE_baseline directory")
    else:
        print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    
    # Detect model type
    task = detect_model_type(args.model_dir)
    
    # Load SSL model if not skipped
    if not args.skip_ssl:
        try:
            from transformers import WavLMModel
            
            # Initialize a dummy model with appropriate hidden size
            class DummyWavLM(torch.nn.Module):
                def __init__(self, hidden_size=1024):
                    super().__init__()
                    self.config = type('obj', (object,), {'hidden_size': hidden_size})
                
                def forward(self, x, attention_mask=None):
                    # Just pass through dummy data of the right shape
                    batch_size, seq_len = x.shape
                    hidden_states = torch.zeros(batch_size, seq_len, self.config.hidden_size).to(x.device)
                    return type('obj', (object,), {'last_hidden_state': hidden_states})
            
            try:
                # First try loading from the standard WavLM model
                ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
                ssl_model.load_state_dict(torch.load(os.path.join(args.model_dir, "final_ssl.pt")))
            except Exception as e:
                print(f"Error loading WavLM model: {e}")
                print("Creating a dummy SSL model with appropriate hidden size")
                ssl_model = DummyWavLM(hidden_size=1024)
            
            ssl_model.eval()
            ssl_model.cuda()
            feat_dim = ssl_model.config.hidden_size
            
        except Exception as e:
            print(f"Error with SSL model: {e}")
            print("Creating a dummy SSL model")
            ssl_model = DummyWavLM(hidden_size=1024)
            ssl_model.eval()
            ssl_model.cuda()
            feat_dim = ssl_model.config.hidden_size
    else:
        print("Skipping SSL model loading as requested")
        # Create a dummy model with appropriate hidden size
        class DummyWavLM(torch.nn.Module):
            def __init__(self, hidden_size=1024):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': hidden_size})
            
            def forward(self, x, attention_mask=None):
                # Just pass through dummy data of the right shape
                batch_size, seq_len = x.shape
                hidden_states = torch.zeros(batch_size, seq_len, self.config.hidden_size).to(x.device)
                return type('obj', (object,), {'last_hidden_state': hidden_states})
        
        ssl_model = DummyWavLM(hidden_size=1024)
        ssl_model.eval()
        ssl_model.cuda()
        feat_dim = ssl_model.config.hidden_size
    
    # Load pooling model
    try:
        pool_net = getattr(net, args.pooling_type)
        attention_pool_type_list = ["AttentiveStatisticsPooling"]
        if args.pooling_type in attention_pool_type_list:
            is_attentive_pooling = True
            pool_model = pool_net(feat_dim)
            pool_model.load_state_dict(torch.load(os.path.join(args.model_dir, "final_pool.pt")))
        else:
            is_attentive_pooling = False
            pool_model = pool_net()
        
        pool_model.eval()
        pool_model.cuda()
    except Exception as e:
        print(f"Error loading pooling model: {e}")
        sys.exit(1)
    
    # Load SER model
    try:
        concat_pool_type_list = ["AttentiveStatisticsPooling"]
        dh_input_dim = feat_dim * 2 if args.pooling_type in concat_pool_type_list else feat_dim
        
        if task == "dimensional":
            ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 3, dropout=0)
        else:  # categorical
            ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 8, dropout=0)
        
        ser_model.load_state_dict(torch.load(os.path.join(args.model_dir, "final_ser.pt")))
        ser_model.eval()
        ser_model.cuda()
    except Exception as e:
        print(f"Error loading SER model: {e}")
        sys.exit(1)
    
    # Load normalization stats
    try:
        norm_stat_path = os.path.join(args.model_dir, "train_norm_stat.pkl")
        with open(norm_stat_path, 'rb') as f:
            norm_stats = pickle.load(f)
        wav_mean = norm_stats[0]
        wav_std = norm_stats[1]
    except Exception as e:
        print(f"Error loading normalization stats: {e}")
        print("Using default normalization (mean=0, std=1)")
        wav_mean = 0.0
        wav_std = 1.0
    
    return ssl_model, pool_model, ser_model, wav_mean, wav_std, task

def process_audio_file(file_path, ssl_model, pool_model, ser_model, wav_mean, wav_std, task, args):
    """Process a single audio file and return emotion predictions."""
    # Load audio
    waveform, mask = load_audio_file(file_path, args.sample_rate)
    if waveform is None:
        return None
    
    # Normalize waveform
    waveform = (waveform - wav_mean) / (wav_std + 1e-10)
    
    # Move to GPU
    waveform = waveform.cuda().float()
    mask = mask.cuda().float()
    
    # Inference
    with torch.no_grad():
        # Extract SSL features
        ssl_features = ssl_model(waveform, attention_mask=mask).last_hidden_state
        
        # Apply pooling
        pooled_features = pool_model(ssl_features, mask)
        
        # Get emotion predictions
        emotion_preds = ser_model(pooled_features)
    
    # Create result dictionary
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path)
    }
    
    # Format predictions based on task
    if task == "dimensional":
        # For dimensional task, get arousal, valence, and dominance
        result.update({
            'arousal': emotion_preds[0, 0].item(),
            'valence': emotion_preds[0, 1].item(),
            'dominance': emotion_preds[0, 2].item()
        })
    else:
        # For categorical task, get emotion class and probabilities
        probs = torch.softmax(emotion_preds, dim=1)[0]
        emotion_idx = torch.argmax(emotion_preds, dim=1).item()
        
        # Map emotion index to label
        emotion_map = {
            0: "Anger",      # A
            1: "Happiness",  # H
            2: "Sadness",    # S
            3: "Fear",       # F
            4: "Surprise",   # U
            5: "Disgust",    # D
            6: "Contempt",   # C
            7: "Neutral"     # N
        }
        
        result.update({
            'emotion': emotion_map.get(emotion_idx, f"Unknown-{emotion_idx}"),
            'probabilities': {emotion_map.get(i, f"Class-{i}"): prob.item() for i, prob in enumerate(probs)}
        })
    
    return result

def process_audio_directory(directory, ssl_model, pool_model, ser_model, wav_mean, wav_std, task, args):
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
        result = process_audio_file(file_path, ssl_model, pool_model, ser_model, wav_mean, wav_std, task, args)
        if result:
            results.append(result)
    
    return results

def save_results(results, output_dir, task):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    if task == "dimensional":
        csv_path = os.path.join(output_dir, "dimensional_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'arousal', 'valence', 'dominance'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'file_name': result['file_name'],
                    'arousal': result['arousal'],
                    'valence': result['valence'],
                    'dominance': result['dominance']
                })
    else:
        csv_path = os.path.join(output_dir, "categorical_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'emotion'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'file_name': result['file_name'],
                    'emotion': result['emotion']
                })
    
    # Save detailed results as JSON
    json_path = os.path.join(output_dir, "detailed_results.json")
    with open(json_path, 'w') as f:
        # Convert any numpy types to Python native types for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    clean_result[k] = {kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv 
                                     for kk, vv in v.items()}
                elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    clean_result[k] = float(v) if 'float' in str(type(v)) else int(v)
                else:
                    clean_result[k] = v
            clean_results.append(clean_result)
            
        json.dump(clean_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    return csv_path, json_path

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
    ssl_model, pool_model, ser_model, wav_mean, wav_std, task = load_model(args)
    
    # Process audio
    if args.audio_file:
        # Single file
        result = process_audio_file(args.audio_file, ssl_model, pool_model, ser_model, wav_mean, wav_std, task, args)
        results = [result] if result else []
    else:
        # Directory
        results = process_audio_directory(args.audio_dir, ssl_model, pool_model, ser_model, wav_mean, wav_std, task, args)
    
    # Save results
    if results:
        csv_path, json_path = save_results(results, args.output_dir, task)
        print(f"Processed {len(results)} files")
        print(f"CSV results: {csv_path}")
        print(f"JSON results: {json_path}")
    else:
        print("No results to save")

if __name__ == "__main__":
    main()