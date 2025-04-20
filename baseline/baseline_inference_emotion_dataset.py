import os
import sys
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import argparse
from transformers import AutoModelForAudioClassification
from torch.utils.data import DataLoader
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import emotion dataset from your repository
from src.data.emotion_dataset import EmotionDataset, collate_fn_normalize_for_baseline

def parse_args():
    parser = argparse.ArgumentParser(description="Audio emotion prediction using pretrained model")
    parser.add_argument("--model_name", type=str, default="3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes",
                        help="Name of the pretrained model")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to directory containing audio files")
    parser.add_argument("--audio_list", type=str, default=None,
                        help="Path to file with list of audio files and ground truth")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="Path to MSP-Podcast labels file (used with EmotionDataset)")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to use (Train, Development, Test1, Test2, Test3)")
    parser.add_argument("--output_file", type=str, default="emotion_predictions.csv",
                        help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for dataloader")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU for inference if available")
    parser.add_argument("--use_dataloader", action="store_true", default=False,
                        help="Use EmotionDataset and DataLoader for processing")
    parser.add_argument("--add_noise", action="store_true", default=False,
                        help="add noise for audio files")
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

def process_audio_file(file_path, model, mean, std, device):
    """Process a single audio file for emotion prediction."""
    try:
        # Load and normalize audio
        raw_wav, sr = librosa.load(file_path, sr=model.config.sampling_rate)
        norm_wav = (raw_wav - mean) / (std + 0.000001)
        
        # Convert to tensor and move to device (CPU or GPU)
        wavs = torch.tensor(norm_wav).unsqueeze(0).to(device)
        mask = torch.ones(1, len(norm_wav)).to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(wavs, mask)
            logits = pred.logits if hasattr(pred, 'logits') else pred
            
        # Process predictions
        emotion_map = {
            0: 'A',  # Anger
            1: 'S',  # Sadness
            2: 'H',  # Happiness
            3: 'U',  # Surprise
            4: 'F',  # Fear
            5: 'D',  # Disgust
            6: 'C',  # Contempt (if available)
            7: 'N'   # Neutral
        }
        
        # Get softmax probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Get top prediction
        top_idx = torch.argmax(logits, dim=1).item()
        predicted_emotion = emotion_map.get(top_idx, f"Unknown-{top_idx}")
        
        # Create result dictionary
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'predicted_emotion': predicted_emotion,
            'confidence': probs[top_idx].item()
        }
        
        # Add all probabilities to result
        for idx, prob in enumerate(probs):
            if idx in emotion_map:
                result[f'{emotion_map[idx]}_prob'] = prob.item()
        
        return result
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_with_dataloader(model, dataloader, device):
    """Process audio files using EmotionDataset and DataLoader following exactly the example code."""
    results = []
    
    # Get emotion mapping from EmotionDataset
    idx_to_emotion = {v: k for k, v in EmotionDataset.VALID_EMOTIONS_MAP.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Process each sample in the batch individually
            batch_size = len(batch["waveform"])
            
            for j in range(batch_size):
                try:
                    # Extract individual waveform
                    raw_wav = batch["waveform"][j].numpy()
                    
                    # Get file path if available
                    if hasattr(dataloader.dataset, 'samples') and j < len(dataloader.dataset.samples):
                        try:
                            file_path = dataloader.dataset.samples[j]['file_path']
                        except (TypeError, KeyError):
                            file_path = f"sample_{j}"
                    else:
                        file_path = f"sample_{j}"
                    
                    file_name = os.path.basename(file_path)
                    
                    # Normalize exactly as in the example
                    norm_wav = (raw_wav - model.config.mean) / (model.config.std + 0.000001)
                    
                    # Generate mask exactly as in the example
                    mask = torch.ones(1, len(norm_wav)).to(device)
                    
                    # Batch it exactly as in the example
                    wavs = torch.tensor(norm_wav).unsqueeze(0).to(device)
                    
                    # Predict
                    pred = model(wavs, mask)
                    
                    # Get probabilities
                    probs = torch.nn.functional.softmax(pred, dim=1)[0]
                    
                    # Get predicted class
                    pred_idx = torch.argmax(pred, dim=1).item()
                    predicted_emotion = idx_to_emotion.get(pred_idx, f"Unknown-{pred_idx}")
                    
                    # Create result dictionary
                    result = {
                        'file_path': file_path,
                        'file_name': file_name,
                        'predicted_emotion': predicted_emotion,
                        'confidence': probs[pred_idx].item()
                    }
                    
                    # Add ground truth if available
                    if 'C' in batch:
                        gt_idx = batch['C'][j].item()
                        if gt_idx >= 0:  # Valid ground truth
                            gt_emotion = idx_to_emotion.get(gt_idx, f"Unknown-{gt_idx}")
                            result['gt_emotion'] = gt_emotion
                    
                    # Add dimensional values if available
                    for dim in ['A', 'V', 'D']:
                        if dim in batch:
                            result[f'gt_{dim.lower()}'] = batch[dim][j].item()
                    
                    # Add all probabilities
                    for idx, emotion in idx_to_emotion.items():
                        if idx < len(probs):
                            result[f'{emotion}_prob'] = probs[idx].item()
                    
                    results.append(result)
                
                except Exception as e:
                    print(f"Error processing sample {j}: {e}")
                    continue
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return results

#TODO: remove this
def process_with_dataloader_with_normalize(model, dataloader, device, mean, std):
    """Process audio files using EmotionDataset and DataLoader."""
    results = []
    
    # Get emotion mapping from EmotionDataset
    idx_to_emotion = {v: k for k, v in EmotionDataset.VALID_EMOTIONS_MAP.items()}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get input values
            raw_waveforms = batch["waveform"]
            
            batch_size = raw_waveforms.size(0)
            processed_logits = []
            
            # Process each sample in the batch individually to handle variable lengths
            for j in range(batch_size):
                # Get individual waveform
                raw_wav = raw_waveforms[j].numpy()
                
                # Normalize using model's mean/std exactly as in the example
                norm_wav = (raw_wav - mean) / (std + 0.000001)
                
                # Generate mask
                mask = torch.ones(1, len(norm_wav)).to(device)
                
                # Batch it (add dim)
                wavs = torch.tensor(norm_wav).unsqueeze(0).to(device)
                
                # Predict
                pred = model(wavs, mask)
                logits = pred.logits if hasattr(pred, 'logits') else pred
                
                processed_logits.append(logits)
            
            # Stack all logits
            if processed_logits:
                all_logits = torch.cat(processed_logits, dim=0)
                
                # Get softmax probabilities
                probs = torch.nn.functional.softmax(all_logits, dim=1)
                
                # Get predicted emotions
                pred_indices = torch.argmax(all_logits, dim=1)
                
                # Process each item in the batch
                for j in range(batch_size):
                    # Get file path - accessing the dataset directly to get file path
                    sample_idx = i * dataloader.batch_size + j
                    if sample_idx < len(dataloader.dataset):
                        try:
                            file_path = dataloader.dataset.samples[sample_idx]['file_path']
                        except (IndexError, KeyError, TypeError):
                            # Different ways to access file paths depending on dataset implementation
                            try:
                                file_path = dataloader.dataset.samples[sample_idx]
                            except:
                                file_path = f"sample_{i}_{j}"
                        
                        file_name = os.path.basename(file_path)
                    else:
                        file_name = f"sample_{i}_{j}"
                        file_path = file_name  # Fallback
                    
                    # Get predicted emotion
                    pred_idx = pred_indices[j].item()
                    predicted_emotion = idx_to_emotion.get(pred_idx, f"Unknown-{pred_idx}")
                    
                    # Create result dictionary
                    result = {
                        'file_path': file_path,
                        'file_name': file_name,
                        'predicted_emotion': predicted_emotion,
                        'confidence': probs[j, pred_idx].item()
                    }
                    
                    # Add ground truth if available
                    if 'C' in batch:  # Categorical emotion
                        gt_idx = batch['C'][j].item()
                        if gt_idx >= 0:  # Valid ground truth
                            gt_emotion = idx_to_emotion.get(gt_idx, f"Unknown-{gt_idx}")
                            result['gt_emotion'] = gt_emotion
                    
                    # Add dimensional values if available
                    for dim in ['A', 'V', 'D']:  # Arousal, Valence, Dominance
                        if dim in batch:
                            result[f'gt_{dim.lower()}'] = batch[dim][j].item()
                    
                    # Add all probabilities
                    for idx, emotion in idx_to_emotion.items():
                        if idx < probs.shape[1]:
                            result[f'{emotion}_prob'] = probs[j, idx].item()
                    
                    results.append(result)
            
            # Clear GPU cache periodically
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return results

#TODO: remove this
def process_with_dataloader_with_collate_fn(model, dataloader, device):
    """Process audio files using EmotionDataset and DataLoader with custom normalization."""
    results = []
    
    # Get emotion mapping from EmotionDataset
    idx_to_emotion = {v: k for k, v in EmotionDataset.VALID_EMOTIONS_MAP.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Get normalized waveforms and masks from our custom collate function
            normalized_waveforms = [waveform.to(device) for waveform in batch['normalized_waveform']]
            masks = [mask.to(device) for mask in batch['mask']]
            
            # Process each sample (we still need to do this individually due to variable lengths)
            batch_size = len(normalized_waveforms)
            processed_logits = []
            
            for j in range(batch_size):
                # Get individual waveform and mask
                wavs = normalized_waveforms[j].unsqueeze(0)  # Add batch dimension
                mask = masks[j].unsqueeze(0)  # Add batch dimension
                
                # Predict
                pred = model(wavs, mask)
                logits = pred.logits if hasattr(pred, 'logits') else pred
                
                processed_logits.append(logits)
            
            # Stack all logits
            if processed_logits:
                all_logits = torch.cat(processed_logits, dim=0)
                
                # Get softmax probabilities
                probs = torch.nn.functional.softmax(all_logits, dim=1)
                
                # Get predicted emotions
                pred_indices = torch.argmax(all_logits, dim=1)
                
                # Process each item in the batch
                for j in range(batch_size):
                    # Get file path
                    file_path = batch['file_path'][j]
                    file_name = os.path.basename(file_path)
                    
                    # Get predicted emotion
                    pred_idx = pred_indices[j].item()
                    predicted_emotion = idx_to_emotion.get(pred_idx, f"Unknown-{pred_idx}")
                    
                    # Create result dictionary
                    result = {
                        'file_path': file_path,
                        'file_name': file_name,
                        'predicted_emotion': predicted_emotion,
                        'confidence': probs[j, pred_idx].item()
                    }
                    
                    # Add ground truth if available
                    if 'C' in batch:  # Categorical emotion
                        gt_idx = batch['C'][j]
                        if isinstance(gt_idx, torch.Tensor):
                            gt_idx = gt_idx.item()
                        if gt_idx >= 0:  # Valid ground truth
                            gt_emotion = idx_to_emotion.get(gt_idx, f"Unknown-{gt_idx}")
                            result['gt_emotion'] = gt_emotion
                    
                    # Add dimensional values if available
                    for dim in ['A', 'V', 'D']:  # Arousal, Valence, Dominance
                        if dim in batch:
                            dim_val = batch[dim][j]
                            if isinstance(dim_val, torch.Tensor):
                                dim_val = dim_val.item()
                            result[f'gt_{dim.lower()}'] = dim_val
                    
                    # Add all probabilities
                    for idx, emotion in idx_to_emotion.items():
                        if idx < probs.shape[1]:
                            result[f'{emotion}_prob'] = probs[j, idx].item()
                    
                    results.append(result)
            
            # Clear GPU cache periodically
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return results

#TODO: remove this
def process_with_dataloader_bu(model, dataloader, device):
    """Process audio files using EmotionDataset and DataLoader."""
    results = []
    
    # Get emotion mapping from EmotionDataset
    idx_to_emotion = {v: k for k, v in EmotionDataset.VALID_EMOTIONS_MAP.items()}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get input values and ground truth
            input_values = batch["waveform"].to(device)
            
            # For creating mask (all ones with same length as input)
            batch_size, seq_len = input_values.shape
            mask = torch.ones(batch_size, seq_len).to(device)
            
            # Get predictions
            pred = model(input_values, mask)
            logits = pred.logits if hasattr(pred, 'logits') else pred
            
            # Get softmax probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted emotions
            pred_indices = torch.argmax(logits, dim=1)
            
            # Process each item in the batch
            for j in range(batch_size):
                # Get file path - accessing the dataset directly to get file path
                sample_idx = i * dataloader.batch_size + j
                if sample_idx < len(dataloader.dataset):
                    file_path = dataloader.dataset.samples[sample_idx]['file_path']
                    file_name = os.path.basename(file_path)
                else:
                    file_name = f"sample_{i}_{j}"
                    file_path = file_name  # Fallback
                
                # Get predicted emotion
                pred_idx = pred_indices[j].item()
                predicted_emotion = idx_to_emotion.get(pred_idx, f"Unknown-{pred_idx}")
                
                # Create result dictionary
                result = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'predicted_emotion': predicted_emotion,
                    'confidence': probs[j, pred_idx].item()
                }
                
                # Add ground truth if available
                if 'C' in batch:  # Categorical emotion
                    gt_idx = batch['C'][j].item()
                    if gt_idx >= 0:  # Valid ground truth
                        gt_emotion = idx_to_emotion.get(gt_idx, f"Unknown-{gt_idx}")
                        result['gt_emotion'] = gt_emotion
                
                # Add dimensional values if available
                for dim in ['A', 'V', 'D']:  # Arousal, Valence, Dominance
                    if dim in batch:
                        result[f'gt_{dim.lower()}'] = batch[dim][j].item()
                
                # Add all probabilities
                for idx, emotion in idx_to_emotion.items():
                    if idx < probs.shape[1]:
                        result[f'{emotion}_prob'] = probs[j, idx].item()
                
                results.append(result)
            
            # Clear GPU cache periodically
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return results


def main():
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
    
    mean = audio_model.config.mean
    std = audio_model.config.std
    
    # Process audio files
    results = []
    
    if args.use_dataloader and args.labels_file:
        # Use EmotionDataset and DataLoader
        print("Using EmotionDataset for processing")
        
        # Create dataset
        dataset = EmotionDataset(
            labels_file=args.labels_file,
            audio_dir=args.audio_dir,
            split=args.split,
            feature_extractor=None,  # We'll use our own preprocessing
            sample_rate=audio_model.config.sampling_rate,
            categorical_only=False,  # Include all samples
            add_noise=args.add_noise,
            noise_dir="/proj/speech/projects/noise_robustness/Audioset/Audioset-train"
        )
        
        # Create dataloader
        # collate_fn = partial(collate_fn_normalize_for_baseline, mean=mean, std=std)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            # collate_fn=collate_fn
        )
        
        # Process with dataloader
        results = process_with_dataloader(audio_model, dataloader, device)
        # results = process_with_dataloader(audio_model, dataloader, device, mean, std)

        
    elif args.audio_list:
        # Load and process audio list
        audio_files, labels = load_audio_list(args.audio_list, args.audio_dir)
        
        # Process each file
        for i, file_path in enumerate(tqdm(audio_files, desc="Processing audio files")):
            result = process_audio_file(file_path, audio_model, mean, std, device)
            if result:
                # Add ground truth if available
                if i < len(labels):
                    for key, value in labels[i].items():
                        result[f'gt_{key}'] = value
                
                results.append(result)
            
            # Clear GPU cache periodically
            if i % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        print("Error: Must specify either --audio_list or (--labels_file and --use_dataloader)")
        return
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")
    
    # Print summary statistics
    if 'predicted_emotion' in df.columns and 'gt_emotion' in df.columns:
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