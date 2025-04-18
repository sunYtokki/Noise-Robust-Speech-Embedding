import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor

from src.data.augment import add_noise_to_speech
from src.utils.logging_utils import logger
from src.utils.audio_utils import load_and_process_audio, get_audio_files

class EmotionDataset(Dataset):
    VALID_EMOTIONS_MAP = {
            'A': 0,  # Anger
            'H': 1,  # Happiness
            'S': 2,  # Sadness
            'F': 3,  # Fear
            'U': 4,  # Surprise
            'D': 5,  # Disgust
            'C': 6,  # Contempt
            'N': 7,  # Neutral
        }
    INVALID_EMOTIONS_MAP = {
            'X': 8,  # No agreement
            'O': 9,  # Other
        }
    EMOTIONS_MAP= {
        **VALID_EMOTIONS_MAP,
        **INVALID_EMOTIONS_MAP,
    }

    def __init__(self, 
                 labels_file="/proj/speech/projects/noise_robustness/MSP-PODCAST-Publish-1.11/Labels/labels_consensus.csv",
                 audio_dir="/proj/speech/projects/noise_robustness/MSP-PODCAST-Publish-1.11/Audios",
                 noise_dir=None,
                 split=None,  # Can be "Train", "Development", "Test1", "Test2", "Test3" or None to use all
                 feature_extractor=None,
                 sample_rate=16000,
                 max_audio_length=5.0,
                 add_noise=False,
                 snr_range=None,
                 categorical_only=True):
        """
        Dataset for emotion recognition.
        
        Args:
            labels_file: Path to CSV file with emotion labels
            audio_dir: Root directory for audio files if paths in labels are relative
            noise_dir: Path to directory containing noise files
            split: Dataset split to use ("Train", "Development", "Test1", "Test2", "Test3"), or None for all data
            feature_extractor: WavLM feature extractor
            sample_rate: Target sample rate
            max_audio_length: Maximum audio length in seconds
            add_noise: Whether to add noise during training
            snr_range: Range of SNR values for noise augmentation
            categorical_only: If True, only use samples with valid emotion categories
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.feature_extractor = feature_extractor
        self.add_noise = add_noise
        self.categorical_only = categorical_only # Whether to exclude "X" (no agreement) or "O" (other) if categorical_only
        if categorical_only:
            self.emotion_mapping = self.VALID_EMOTIONS_MAP
        else:
            self.emotion_mapping = self.EMOTIONS_MAP
        # self.num_classes = len(self.emotion_mapping.keys())
        self.idx_to_emotion = {v: k for k, v in self.emotion_mapping.items()}
        
        # Parse labels file
        self.df = pd.read_csv(labels_file)
        
        # Filter by split if specified
        if split:
            self.df = self.df[self.df['Split_Set'] == split]
            
        # Build df with classes
        self.df = self.df[self.df['EmoClass'].isin(self.emotion_mapping.keys())]
        
        # Convert to list of dictionaries for easier access
        self.samples = []
        skipped_samples = 0
        unknown_categories = set()
        
        for _, row in self.df.iterrows():
            file_name = row['FileName']
            category = row['EmoClass']
            
            # Check if category is in mapping
            if category not in self.emotion_mapping:
                unknown_categories.add(category)
                skipped_samples += 1
                continue
            
            # Build file path
            file_path = os.path.join(audio_dir, file_name) if audio_dir else file_name
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Audio file not found: {file_path}")
                skipped_samples += 1
                continue
            
            # Get dimensional attributes
            arousal = float(row['EmoAct'])
            valence = float(row['EmoVal'])
            dominance = float(row['EmoDom'])
            
            # Add additional metadata if available
            metadata = {}
            if 'SpkrID' in row:
                metadata['speaker_id'] = row['SpkrID']
            if 'Gender' in row:
                metadata['gender'] = row['Gender']
            
            self.samples.append({
                'file_path': file_path,
                'category': category,
                'category_idx': self.emotion_mapping.get(category, -1),
                'arousal': arousal,
                'valence': valence,
                'dominance': dominance,
                'metadata': metadata
            })
        
        logger.info(f"Loaded {len(self.samples)} valid samples from {labels_file}")
        if split:
            logger.info(f"Using split: {split}")
        if skipped_samples > 0:
            logger.info(f"Skipped {skipped_samples} invalid samples")
        if unknown_categories:
            logger.warning(f"Found unknown emotion categories: {unknown_categories}")
        
        # Load noise files if needed
        if add_noise and noise_dir:
            self.noise_files = get_audio_files(noise_dir)
            self.snr_range = snr_range or [0, 5, 10, 15, 20]
            logger.info(f"Found {len(self.noise_files)} noise files for augmentation")
        
        # Log category distribution
        self._log_category_distribution()
    
    def _log_category_distribution(self):
        """Log the distribution of emotion categories."""
        from collections import Counter
        category_counts = Counter([sample['category'] for sample in self.samples])
        total_samples = len(self.samples)
        
        logger.info("Category distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / total_samples) * 100
            logger.info(f"  {category}: {count} samples ({percentage:.2f}%)")
    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load and process audio
        waveform = self._load_and_process_audio(sample['file_path'])
        
        # Add noise if specified
        if self.add_noise and hasattr(self, 'noise_files') and len(self.noise_files) > 0:
            # Select random noise file
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise = self._load_and_process_audio(self.noise_files[noise_idx])
            
            # Select random SNR
            snr = random.choice(self.snr_range)
            
            # Add noise to speech
            noisy_waveform = add_noise_to_speech(waveform, noise, snr)
            
            # If noise addition failed, use original waveform
            if noisy_waveform is None:
                noisy_waveform = waveform
        else:
            noisy_waveform = waveform
        
        # Process with feature extractor
        if self.feature_extractor:
            inputs = self.feature_extractor(
                noisy_waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            return {
                "input_values": inputs.input_values.squeeze(0),
                "C": sample['category_idx'],
                "A": sample['arousal'],
                "V": sample['valence'],
                "D": sample['dominance']
            }
        else:
            return {
                "waveform": noisy_waveform.squeeze(0),
                "C": sample['category_idx'],
                "A": sample['arousal'],
                "V": sample['valence'],
                "D": sample['dominance']
            }
    
    def _load_and_process_audio(self, file_path):
        """Load and process audio file."""
        waveform = load_and_process_audio(
            file_path, 
            sample_rate=self.sample_rate,
            max_audio_length=self.max_samples/self.sample_rate,
            random_crop=True
        )
        
        if waveform is None:
            # TODO: verify if this is valid way
            # Return a small non-zero tensor to avoid crashes
            logger.warning(f"Returning fallback tensor for {file_path}")
            return torch.ones((1, self.max_samples)) * 1e-6
        
        return waveform


def create_emotion_dataloaders(config, feature_extractor):
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = EmotionDataset(
        # labels_file=config['emotion']['train_labels_file'],
        # audio_dir=config['emotion']['train_data_path'],
        split=config['emotion']['train_dataset_split'],
        feature_extractor=feature_extractor,
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        add_noise=config['emotion']['add_noise_during_training'],
        noise_dir=config['data']['noise_data_path'],
        snr_range=config['data']['snr_range'],
        categorical_only=config['emotion']['categorical_only']
    )
    
    val_dataset = EmotionDataset(
        # labels_file=config['emotion']['val_labels_file'],
        # audio_dir=config['emotion']['val_data_path'],
        split=config['emotion']['validataion_dataset_split'],
        feature_extractor=feature_extractor,
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        add_noise=config['emotion']['add_noise_during_training'],
        noise_dir=config['data']['noise_data_path'],
        snr_range=config['data']['snr_range'],
        categorical_only=config['emotion']['categorical_only']
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['emotion']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['emotion']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    return train_loader, val_loader