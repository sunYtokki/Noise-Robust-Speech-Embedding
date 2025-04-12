import os
from tqdm import tqdm
import torch, torchaudio

def preprocess_audio_files(directory, min_duration=0.1):
    """
    Scan audio files and identify problematic ones
    
    Args:
        directory: Directory containing audio files
        min_duration: Minimum valid duration in seconds
        
    Returns:
        valid_files: List of valid audio file paths
        invalid_files: List of invalid audio file paths
    """
    valid_files = []
    invalid_files = []
    
    extensions = {'.wav', '.flac', '.mp3'}
    file_paths = [
        os.path.join(root, f) 
        for root, _, files in os.walk(directory) 
        for f in files if os.path.splitext(f)[1].lower() in extensions
    ]
    
    for file_path in tqdm(file_paths, desc="Validating audio files"):
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Check duration
            duration = waveform.shape[1] / sr
            if duration < min_duration:
                print(f"File too short: {file_path} ({duration:.2f}s)")
                invalid_files.append(file_path)
                continue
                
            if check_abnormal_values(waveform):
                print(f"Invalid audio in: {file_path}")
                invalid_files.append(file_path)
                continue
                
            valid_files.append(file_path)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            invalid_files.append(file_path)
            
    print(f"Found {len(valid_files)} valid files and {len(invalid_files)} invalid files")
    return valid_files, invalid_files

def check_abnormal_values(waveform):
    return torch.isnan(waveform).any() or torch.isinf(waveform).any() or (waveform.abs().sum() < 1e-6)