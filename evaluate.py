from src.models.byol import BYOLSpeechModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def evaluate_embedding_similarity(model: BYOLSpeechModel, 
                                 dataloader: DataLoader, 
                                 device: torch.device,
                                 num_samples: int = 100) -> dict:
    """
    Evaluate embedding similarity between clean and noisy speech.
    
    Args:
        model: BYOL model
        dataloader: DataLoader for evaluation data
        device: Device to use
        num_samples: Number of samples to evaluate
        
    Returns:
        similarities: Dictionary of SNR to average similarity
    """
    model.eval()
    encoder = model.get_encoder()
    
    # Create a dictionary to store similarities for each SNR value
    similarities = {snr: [] for snr in dataloader.dataset.snr_range}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples // dataloader.batch_size:
                break
                
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            
            # Get batch SNR values - this is a tensor with batch_size elements
            batch_snrs = batch["snr"].tolist()  # Convert to Python list
            
            # Get embeddings
            clean_emb = encoder(clean_input_values)
            noisy_emb = encoder(noisy_input_values)
            
            # Normalize embeddings
            clean_emb = F.normalize(clean_emb, dim=1)
            noisy_emb = F.normalize(noisy_emb, dim=1)
            
            # Compute cosine similarity for each item in batch
            # This computes similarity for each pair of clean/noisy embeddings
            similarity = torch.sum(clean_emb * noisy_emb, dim=1)
            
            # Add each similarity value to the corresponding SNR list
            for idx, snr in enumerate(batch_snrs):
                similarities[snr].append(similarity[idx].item())
    
    # Compute average similarity for each SNR
    avg_similarities = {snr: sum(sims) / len(sims) if len(sims) > 0 else 0 
                        for snr, sims in similarities.items()}
    
    return avg_similarities


def visualize_embeddings(model: BYOLSpeechModel, 
                         dataloader: DataLoader, 
                         device: torch.device,
                         log_dir: str,
<<<<<<< HEAD
                         epoch: int,
=======
                         img_file_name: str,
>>>>>>> dev
                         num_samples: int = 100):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        model: BYOL model
        dataloader: DataLoader for visualization data
        device: Device to use
        log_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    model.eval()
    encoder = model.get_encoder()
    
    clean_embeddings = []
    noisy_embeddings = []
    snrs = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples // dataloader.batch_size:
                break
                
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            
            # Get batch SNR values - convert to a list instead of trying to get a single item
            batch_snrs = batch["snr"].tolist()
            
            # Get embeddings
            clean_emb = encoder(clean_input_values).cpu().numpy()
            noisy_emb = encoder(noisy_input_values).cpu().numpy()
            
            # Add embeddings and SNRs to lists
            clean_embeddings.append(clean_emb)
            noisy_embeddings.append(noisy_emb)
            snrs.extend(batch_snrs)  # Extend the list with all SNRs in the batch
    
    # Concatenate embeddings
    clean_embeddings = np.concatenate(clean_embeddings, axis=0)
    noisy_embeddings = np.concatenate(noisy_embeddings, axis=0)
    
    # Combine embeddings
    all_embeddings = np.concatenate([clean_embeddings, noisy_embeddings], axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(all_embeddings)
    
    # Split back into clean and noisy
    clean_tsne = tsne_embeddings[:len(clean_embeddings)]
    noisy_tsne = tsne_embeddings[len(clean_embeddings):]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot clean embeddings
    plt.scatter(clean_tsne[:, 0], clean_tsne[:, 1], marker='o', color='blue', alpha=0.7, label='Clean')
    
    # Get unique SNR values
    unique_snrs = sorted(list(set(snrs)))
    
    # Plot noisy embeddings with different colors based on SNR
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_snrs)))
    for i, snr in enumerate(unique_snrs):
        indices = [j for j, s in enumerate(snrs) if s == snr]
        if indices:
            plt.scatter(
                noisy_tsne[indices, 0], 
                noisy_tsne[indices, 1], 
                marker='x', 
                color=colors[i], 
                alpha=0.7, 
                label=f'Noisy (SNR={snr}dB)'
            )
    
    plt.title('t-SNE Visualization of Speech Embeddings')
    plt.legend()
    plt.savefig(os.path.join(log_dir, img_file_name))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(log_dir, img_file_name)}")
