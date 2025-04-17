# src/utils/audio_utils.py
import torch
import torchaudio
import random
import os

from src.utils.logging_utils import logger

def load_and_process_audio(file_path, sample_rate=16000, max_audio_length=5.0, random_crop=True):
    """
    Load and process audio file to a standardized format.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        max_audio_length: Maximum audio length in seconds
        random_crop: Whether to randomly crop (True) or take the first segment (False)
        
    Returns:
        waveform: Processed audio waveform tensor [1, max_samples]
    """
    try:
        # Calculate maximum number of samples
        max_samples = int(max_audio_length * sample_rate)
        
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        if waveform.shape[1] > max_samples:
            # Either randomly crop or take the first segment
            if random_crop:
                start = random.randint(0, waveform.shape[1] - max_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + max_samples]
        elif waveform.shape[1] < max_samples:
            # Pad with zeros
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Validate output
        if torch.isnan(waveform).any():
            logger.warning(f"NaN values detected in processed audio: {file_path}")
            return None
            
        if torch.max(torch.abs(waveform)) < 1e-8:
            logger.warning(f"Audio values too small (near zero): {file_path}")
            return None
            
        return waveform
        
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return None


def get_audio_files(directory):
    """Get all audio files from a directory."""
    extensions = {'.wav', '.flac', '.mp3'}
    return [
        os.path.join(root, f) 
        for root, _, files in os.walk(directory) 
        for f in files if os.path.splitext(f)[1].lower() in extensions
    ]