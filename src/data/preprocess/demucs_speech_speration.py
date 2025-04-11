#!/usr/bin/env python3
import os
import argparse
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model

def process_folder(input_folder, output_folder, model_sr=44100, file_sr=16000, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Process all WAV files in the input folder using Demucs and save extracted speech to output folder.
    
    Args:
        input_folder: Path to folder containing WAV files
        output_folder: Path to save processed files
        model_sr: Sample rate expected by the model
        file_sr: Sample rate to save output files
        device: Device to run inference on (cuda/cpu)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the Demucs model - htdemucs_ft is fine-tuned for better voice separation
    print(f"Loading Demucs model on {device}...")
    model = get_model('htdemucs_ft')
    model.to(device)
    model.eval()
    
    # Get all wav files
    wav_files = list(Path(input_folder).glob("**/*.wav"))
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Process each file
    for wav_path in tqdm(wav_files, desc="Processing files"):
        # Create relative path in output directory
        rel_path = wav_path.relative_to(input_folder)
        output_path = Path(output_folder) / rel_path
        
        # Create subdirectories if needed
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(str(wav_path), sr=None, mono=False)
            
            # Handle mono vs stereo
            if audio_data.ndim == 1:
                # Convert mono to stereo by duplicating
                audio_data = np.stack([audio_data, audio_data])
            elif audio_data.shape[0] > 2:
                # If more than 2 channels, keep only first 2
                audio_data = audio_data[:2]
            
            # Resample if needed using librosa
            if sample_rate != model_sr:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=model_sr,
                    res_type='kaiser_best'
                )
                sample_rate = model_sr
            
            # Convert to PyTorch tensor
            waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
            
            # Add batch dimension
            mixture = waveform.unsqueeze(0)
            
            # Apply model for source separation
            with torch.no_grad():
                sources = apply_model(model, mixture, device=device)
                # Sources will be a tensor with [batch, source, channels, time]
                # For htdemucs_ft, sources are ['drums', 'bass', 'other', 'vocals']
                speech = sources[:, 3]  # Get vocals/speech track (index 3 is 'vocals')
            
            # Remove batch dimension
            speech = speech.squeeze(0)
            
            # Use first channel instead of averaging for stereo to mono conversion
            if speech.shape[0] > 1:
                speech = speech[0:1]  # Just take the first channel
            
            # Apply mild denoising by setting very small values to zero
            noise_floor = 0.005  # Adjust this threshold as needed
            speech = speech * (torch.abs(speech) > noise_floor).float()
            
            # Move to CPU for processing
            speech = speech.cpu().numpy()
            
            # Squeeze to remove channel dimension for mono output
            speech = np.squeeze(speech)
            
            # Resample to target sample rate using librosa if needed
            if sample_rate != file_sr:
                speech = librosa.resample(
                    speech, 
                    orig_sr=sample_rate, 
                    target_sr=file_sr,
                    res_type='kaiser_best'
                )
            
            # Save processed audio using soundfile with higher bit depth
            sf.write(str(output_path), speech, file_sr, subtype='PCM_16')
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
    
    print(f"Processing complete. Separated speech saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate speech from audio files using Demucs")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing WAV files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for processed files")
    parser.add_argument("--model_sr", type=int, default=44100, help="Model sample rate")
    parser.add_argument("--file_sr", type=int, default=16000, help="Output file sample rate") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    process_folder(args.input, args.output, model_sr=args.model_sr, file_sr=args.file_sr, device=args.device)