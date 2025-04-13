import torch
from src.utils.logging_utils import logger 

def add_noise_to_speech(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add noise to speech at a specific SNR level with detailed debugging."""
    # Initial check
    if torch.isnan(speech).any():
        logger.warning("Original speech contains NaN values!")
        return None
        
    if torch.isnan(noise).any():
        logger.warning("Original noise contains NaN values!")
        return None
    
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
    
    logger.debug(f"Speech power: {speech_power.item()}, Noise power: {noise_power.item()}")
    
    # Check for zeros or very small values
    if speech_power < 1e-10:
        logger.warning(f"Speech power too small: {speech_power.item()}")
        return None
        
    if noise_power < 1e-10:
        logger.warning(f"Noise power too small: {noise_power.item()}")
        return None
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = torch.sqrt(speech_power / (noise_power * snr_linear))
    
    logger.debug(f"SNR: {snr_db}dB, Linear: {snr_linear}, Noise scaling: {noise_scaling.item()}")
    
    # Check for extreme scaling values
    if torch.isinf(noise_scaling) or torch.isnan(noise_scaling):
        logger.warning(f"Invalid noise scaling: {noise_scaling.item()}")
        return None
        
    if noise_scaling > 1e6:
        logger.warning(f"Extremely large noise scaling: {noise_scaling.item()}")
        return None
    
    # Scale noise and add to speech
    scaled_noise = noise * noise_scaling
    
    if torch.isnan(scaled_noise).any():
        logger.warning("Scaled noise contains NaN values!")
        return None
        
    noisy_speech = speech + scaled_noise
    
    if torch.isnan(noisy_speech).any():
        logger.warning("Resulting noisy speech contains NaN values!")
        return None
    
    return noisy_speech
