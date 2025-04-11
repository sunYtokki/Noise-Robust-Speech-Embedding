# Noise Augmentation Utilities
import torch

def add_noise_to_speech(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add noise to speech at a specific SNR level.
    
    Args:
        speech: Clean speech tensor [1, T]
        noise: Noise tensor [1, T]
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Noisy speech tensor [1, T]
    """
    # Make sure noise is the same length as speech
    if noise.shape[1] > speech.shape[1]:
        noise = noise[:, :speech.shape[1]]
    elif noise.shape[1] < speech.shape[1]:
        # Repeat noise if needed
        repetitions = (speech.shape[1] // noise.shape[1]) + 1
        noise = noise.repeat(1, repetitions)[:, :speech.shape[1]]
    
    # Calculate speech and noise power
    speech_power = torch.mean(speech ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = torch.sqrt(speech_power / (noise_power * snr_linear))
    
    # Scale noise and add to speech
    scaled_noise = noise * noise_scaling
    noisy_speech = speech + scaled_noise
    
    return noisy_speech
