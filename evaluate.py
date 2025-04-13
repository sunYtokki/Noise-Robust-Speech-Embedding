from src.models.byol import BYOLSpeechModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def evaluate_embedding_similarity(model: BYOLSpeechModel, 
                                 dataloader: DataLoader, 
                                 device: torch.device,
                                 config: dict) -> dict:
    """
    Evaluate embedding similarity between clean and noisy speech on the entire dataset.
    
    Args:
        model: BYOL model
        dataloader: DataLoader for evaluation data
        device: Device to use
        config: Configuration dictionary containing SNR range
        
    Returns:
        similarities: Dictionary of SNR to average similarity
    """
    model.eval()
    encoder = model.get_encoder()
    
    # Get SNR range from config instead of dataset
    snr_range = config['data']['snr_range']
    # Create a dictionary to store similarities for each SNR value
    similarities = {snr: [] for snr in snr_range}
    
    with torch.no_grad():
        # Process all batches in the dataloader
        for batch in tqdm(dataloader, desc="Evaluating similarity"):
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            
            # Get batch SNR values
            batch_snrs = batch["snr"].tolist()  # Convert to Python list
            
            # Get embeddings
            clean_emb = encoder(clean_input_values)
            noisy_emb = encoder(noisy_input_values)
            
            # Normalize embeddings
            clean_emb = F.normalize(clean_emb, dim=1)
            noisy_emb = F.normalize(noisy_emb, dim=1)
            
            # Compute cosine similarity for each item in batch
            similarity = torch.sum(clean_emb * noisy_emb, dim=1)
            
            # Add each similarity value to the corresponding SNR list
            for idx, snr in enumerate(batch_snrs):
                if snr in similarities:
                    similarities[snr].append(similarity[idx].item())
    
    # Compute average similarity for each SNR
    avg_similarities = {snr: sum(sims) / len(sims) if len(sims) > 0 else 0 
                        for snr, sims in similarities.items()}
    
    return avg_similarities


def validate_model(model: BYOLSpeechModel, 
                  val_loader: DataLoader, 
                  device: torch.device,
                  config: dict) -> tuple:
    """
    Validate the model on the entire validation set.
    
    Args:
        model: BYOL model
        val_loader: Validation data loader
        device: Device to use
        config: Configuration dictionary
        
    Returns:
        val_loss: Average validation loss
        metrics: Dictionary of validation metrics
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    # Calculate embedding similarities for validation
    similarities = evaluate_embedding_similarity(model, val_loader, device, config)
    
    # Compute validation loss on the entire validation set
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating validation loss"):
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            
            # Forward pass
            online_pred, target_proj = model(clean_input_values, noisy_input_values)
            
            # Compute loss
            from src.models.byol import byol_loss
            loss = byol_loss(online_pred, target_proj)
            
            val_loss += loss.item()
            num_batches += 1
    
    # Calculate average validation loss
    avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate average similarity across all SNR levels
    avg_similarity = sum(similarities.values()) / len(similarities) if similarities else 0.0
    
    # Prepare metrics dictionary
    metrics = {
        'val_loss': avg_val_loss,
        'val_avg_similarity': avg_similarity,
        'val_similarities': similarities
    }
    
    return avg_val_loss, metrics


def visualize_embeddings(model: BYOLSpeechModel, 
                         dataloader: DataLoader, 
                         device: torch.device,
                         log_dir: str,
                         img_file_name: str,
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