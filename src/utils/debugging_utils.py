import torch

def check_audio_tensor(tensor, name, max_threshold=1e6, min_threshold=1e-6, print_stats=False):
    """Check tensor for abnormal values like NaN, Inf, or extremely large values."""
    if torch.isnan(tensor).any():
        print(f"WARNING: {name} contains NaN values!")
        return False
    
    if torch.isinf(tensor).any():
        print(f"WARNING: {name} contains Inf values!")
        return False
    
    if tensor.abs().sum() < min_threshold:
        print(f"WARNING: {name} contains very small values!")
        return False
    
    max_val = tensor.abs().max().item()
    if max_val > max_threshold:
        print(f"WARNING: {name} contains very large values! Max abs value: {max_val}")
        return False
    
    if print_stats:
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        print(f"Stats for {name}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    return True
