import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List
from transformers import AutoFeatureExtractor

from src.data.augment import add_noise_to_speech
from src.utils.logging_utils import logger
from src.utils.audio_utils import load_and_process_audio, get_audio_files

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
        self.clean_files = get_audio_files(clean_data_path)
        self.noise_files = get_audio_files(noise_data_path)
        
        print(f"Found {len(self.clean_files)} clean files and {len(self.noise_files)} noise files.")
        
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def _load_and_process_audio(self, file_path: str) -> torch.Tensor:
        """Load audio file and process to target sample rate and length."""
        return load_and_process_audio(
            file_path, 
            sample_rate=self.sample_rate,
            max_audio_length=self.max_samples/self.sample_rate,
            random_crop=True
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pair of clean and noisy speech with enhanced error checking."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Load clean speech
            clean_speech = self._load_and_process_audio(self.clean_files[idx])
            
            # If invalid audio, try the next file
            if clean_speech is None:
                logger.warning(f"Invalid clean speech file, trying next (attempt {attempt+1})")
                idx = (idx + 1) % len(self.clean_files)
                continue
            
            # Load random noise
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise = self._load_and_process_audio(self.noise_files[noise_idx])
            
            # If invalid noise, try another noise file
            if noise is None:
                logger.warning(f"Invalid noise file, trying another (attempt {attempt+1})")
                continue
            
            # Select random SNR
            snr = random.choice(self.snr_range)
            
            # Add noise to speech with detailed debugging
            noisy_speech = add_noise_to_speech(clean_speech, noise, snr)
            
            # If noise addition failed, try again
            if noisy_speech is None:
                logger.warning(f"Noise addition failed, trying again (attempt {attempt+1})")
                continue
            
            # Normalize audio
            try:
                clean_max = torch.max(torch.abs(clean_speech))
                noisy_max = torch.max(torch.abs(noisy_speech))
                
                # Check if normalization would cause issues
                if clean_max < 1e-8:
                    logger.warning(f"Clean speech maximum too small: {clean_max.item()}")
                    continue
                    
                if noisy_max < 1e-8:
                    logger.warning(f"Noisy speech maximum too small: {noisy_max.item()}")
                    continue
                
                clean_speech = clean_speech / (clean_max + 1e-8)
                noisy_speech = noisy_speech / (noisy_max + 1e-8)
                
                # Final NaN check after normalization
                if torch.isnan(clean_speech).any():
                    logger.warning("Normalized clean speech contains NaN values!")
                    continue
                    
                if torch.isnan(noisy_speech).any():
                    logger.warning("Normalized noisy speech contains NaN values!")
                    continue
                
            except Exception as e:
                logger.error(f"Error during normalization: {e}")
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
                    logger.warning("Feature-extracted clean input contains NaN values!")
                    continue
                    
                if torch.isnan(noisy_input.input_values).any():
                    logger.warning("Feature-extracted noisy input contains NaN values!")
                    continue
                    
                return {
                    "clean_input_values": clean_input.input_values,
                    "noisy_input_values": noisy_input.input_values,
                    "snr": snr
                }
                
            except Exception as e:
                logger.error(f"Error during feature extraction: {e}")
                continue
    

def create_dataloaders(config, feature_extractor=None):
    """Create training and validation dataloaders from the configuration."""
    # Create the main dataset
    dataset = NoiseRobustSpeechDataset(
        clean_data_path=config['data']['clean_data_path'],
        noise_data_path=config['data']['noise_data_path'],
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        snr_range=config['data']['snr_range'],
        feature_extractor=feature_extractor
    )
    
    # Split dataset into train and validation
    val_ratio = config['data'].get('validation_ratio', 0.1)
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    logger.info(f"Splitting dataset: {train_size} training samples, {val_size} validation samples")
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training'].get('seed', 42))
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,  # No need to shuffle validation data
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader
