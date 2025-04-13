import torch
from torch.utils.data import TensorDataset

def create_mock_dataset(num_samples=100, snr_range=[0, 5, 10, 15, 20]):
    # Create random input tensors
    clean_inputs = torch.randn(num_samples, 10)
    
    # Create noisy versions with different SNR levels
    noisy_inputs = []
    snrs = []
    
    for i in range(num_samples):
        # Randomly select an SNR
        snr = snr_range[i % len(snr_range)]
        snrs.append(snr)
        
        # Create noisy version (more noise for lower SNR)
        noise_level = 1.0 / (snr + 1)
        noise = torch.randn(10) * noise_level
        noisy_inputs.append(clean_inputs[i] + noise)
    
    noisy_inputs = torch.stack(noisy_inputs)
    snrs = torch.tensor(snrs)
    
    return TensorDataset(clean_inputs, noisy_inputs, snrs)
