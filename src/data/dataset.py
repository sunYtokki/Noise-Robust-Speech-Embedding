import os
import torch
import random
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
from transformers import AutoFeatureExtractor

from .augment import add_noise_to_speech

class NoiseRobustSpeechDataset(Dataset):
    def __init__(self, 
                 clean_data_path: str, 
                 noise_data_path: str,
                 sample_rate: int = 16000,
                 max_audio_length: float = 5.0,
                 snr_range: List[int] = [0, 5, 10, 15, 20],
                 feature_extractor = None):
        """
        Dataset for noise-robust speech embedding training
        
        Args:
            clean_data_path: Path to clean speech files
            noise_data_path: Path to noise files
            sample_rate: Target sample rate
            max_audio_length: Maximum audio length in seconds
            snr_range: Range of SNR values for noise augmentation
            feature_extractor: WavLM feature extractor
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.snr_range = snr_range
        self.feature_extractor = feature_extractor
        
        # Load file paths
        self.clean_files = self._get_audio_files(clean_data_path)
        self.noise_files = self._get_audio_files(noise_data_path)
        
        print(f"Found {len(self.clean_files)} clean files and {len(self.noise_files)} noise files.")
        
    def _get_audio_files(self, directory: str) -> List[str]:
        """Get all audio files from a directory."""
        extensions = {'.wav', '.flac', '.mp3'}
        return [
            os.path.join(root, f) 
            for root, _, files in os.walk(directory) 
            for f in files if os.path.splitext(f)[1].lower() in extensions
        ]
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def _load_and_process_audio(self, file_path: str) -> torch.Tensor:
        """Load audio file and process to target sample rate and length."""
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        if waveform.shape[1] > self.max_samples:
            # Randomly crop
            start = random.randint(0, waveform.shape[1] - self.max_samples)
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            # Pad with zeros
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform

    #TODO: clean up logging
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pair of clean and noisy speech with enhanced error checking."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Load clean speech
            clean_speech = self._load_and_process_audio(self.clean_files[idx])
            
            # If invalid audio, try the next file
            if clean_speech is None:
                print(f"Invalid clean speech file, trying next (attempt {attempt+1})")
                idx = (idx + 1) % len(self.clean_files)
                continue
            
            # Load random noise
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise = self._load_and_process_audio(self.noise_files[noise_idx])
            
            # If invalid noise, try another noise file
            if noise is None:
                print(f"Invalid noise file, trying another (attempt {attempt+1})")
                continue
            
            # Select random SNR
            snr = random.choice(self.snr_range)
            
            # Add noise to speech with detailed debugging
            noisy_speech = add_noise_to_speech(clean_speech, noise, snr)
            
            # If noise addition failed, try again
            if noisy_speech is None:
                print(f"Noise addition failed, trying again (attempt {attempt+1})")
                continue
            
            # Normalize audio
            try:
                clean_max = torch.max(torch.abs(clean_speech))
                noisy_max = torch.max(torch.abs(noisy_speech))
                
                # Check if normalization would cause issues
                if clean_max < 1e-8:
                    print(f"Clean speech maximum too small: {clean_max.item()}")
                    continue
                    
                if noisy_max < 1e-8:
                    print(f"Noisy speech maximum too small: {noisy_max.item()}")
                    continue
                
                clean_speech = clean_speech / (clean_max + 1e-8)
                noisy_speech = noisy_speech / (noisy_max + 1e-8)
                
                # Final NaN check after normalization
                if torch.isnan(clean_speech).any():
                    print("Normalized clean speech contains NaN values!")
                    continue
                    
                if torch.isnan(noisy_speech).any():
                    print("Normalized noisy speech contains NaN values!")
                    continue
                
            except Exception as e:
                print(f"Error during normalization: {e}")
                continue
            
            # Process with feature extractor
            try:
                clean_input = self.feature_extractor(
                    clean_speech.squeeze().numpy(), 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                noisy_input = self.feature_extractor(
                    noisy_speech.squeeze().numpy(), 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                # Final NaN check after feature extraction
                if torch.isnan(clean_input.input_values).any():
                    print("Feature-extracted clean input contains NaN values!")
                    continue
                    
                if torch.isnan(noisy_input.input_values).any():
                    print("Feature-extracted noisy input contains NaN values!")
                    continue
                    
                return {
                    "clean_input_values": clean_input.input_values,
                    "noisy_input_values": noisy_input.input_values,
                    "snr": snr
                }
                
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                continue
    

def create_dataloader(config, feature_extractor=None):
    """Create a dataloader from the configuration."""
    dataset = NoiseRobustSpeechDataset(
        clean_data_path=config['data']['clean_data_path'],
        noise_data_path=config['data']['noise_data_path'],
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        snr_range=config['data']['snr_range'],
        feature_extractor=feature_extractor
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader