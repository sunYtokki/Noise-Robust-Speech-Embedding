import torch
import os
from config import get_config
# from utils.train_utils import train_model
import random
import numpy as np
from torch.utils.data import DataLoader
from models.byol import BYOLSpeechModel
from data.dataset import NoiseRobustSpeechDataset
import tqdm
from models.byol import byol_loss
import wandb
from transformers import AutoFeatureExtractor
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from typing import Dict


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Training function
def train_one_epoch(model: BYOLSpeechModel, 
                    dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    device: torch.device) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: BYOL model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move inputs to device
        clean_input_values = batch["clean_input_values"].to(device)
        noisy_input_values = batch["noisy_input_values"].to(device)
        
        # Forward pass
        online_pred, target_proj = model(clean_input_values, noisy_input_values)
        
        # Compute loss
        loss = byol_loss(online_pred, target_proj)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target network with EMA
        model._update_target_network()
        
        # Update learning rate
        scheduler.step()
        
        # Track loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Evaluation function for embedding similarity
def evaluate_embedding_similarity(model: BYOLSpeechModel, 
                                  config,
                                 dataloader: DataLoader, 
                                 device: torch.device,
                                 num_samples: int = 100) -> Dict[int, float]:
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
    
    similarities = {snr: [] for snr in config.snr_range}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            snr = batch["snr"].item()
            
            # Get embeddings
            clean_emb = encoder(clean_input_values)
            noisy_emb = encoder(noisy_input_values)
            
            # Normalize embeddings
            clean_emb = F.normalize(clean_emb, dim=1)
            noisy_emb = F.normalize(noisy_emb, dim=1)
            
            # Compute cosine similarity
            similarity = torch.sum(clean_emb * noisy_emb, dim=1).item()
            similarities[snr].append(similarity)
    
    # Compute average similarity for each SNR
    avg_similarities = {snr: sum(sims) / len(sims) if len(sims) > 0 else 0 
                        for snr, sims in similarities.items()}
    
    return avg_similarities

# Visualization function for t-SNE
def visualize_embeddings(model: BYOLSpeechModel, 
                         config,
                         dataloader: DataLoader, 
                         device: torch.device,
                         num_samples: int = 100):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        model: BYOL model
        dataloader: DataLoader for visualization data
        device: Device to use
        num_samples: Number of samples to visualize
    """
    model.eval()
    encoder = model.get_encoder()
    
    clean_embeddings = []
    noisy_embeddings = []
    snrs = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move inputs to device
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            snr = batch["snr"].item()
            
            # Get embeddings
            clean_emb = encoder(clean_input_values).cpu().numpy()
            noisy_emb = encoder(noisy_input_values).cpu().numpy()
            
            clean_embeddings.append(clean_emb)
            noisy_embeddings.append(noisy_emb)
            snrs.append(snr)
    
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
    
    # Plot noisy embeddings with different colors based on SNR
    colors = plt.cm.rainbow(np.linspace(0, 1, len(config.snr_range)))
    for i, snr in enumerate(config.snr_range):
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
    plt.savefig(os.path.join(config.log_dir, 'tsne_embeddings.png'))
    plt.close()


# Main training loop
def train_model(config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Initialize wandb
    wandb.init(project="noise-robust-speech-embeddings", config=vars(config))
    
    # Set device
    device = torch.device(config.device)
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
    
    # Create dataset and dataloader
    dataset = NoiseRobustSpeechDataset(
        clean_data_path=config.clean_data_path,
        noise_data_path=config.noise_data_path,
        sample_rate=config.sample_rate,
        max_audio_length=config.max_audio_length,
        snr_range=config.snr_range,
        feature_extractor=feature_extractor
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = BYOLSpeechModel(config).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(dataloader) * config.num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss:.4f}")
        
        # Evaluate embedding similarity
        if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
            similarities = evaluate_embedding_similarity(model, dataloader, device)
            
            for snr, sim in similarities.items():
                wandb.log({f"similarity_snr_{snr}": sim})
            
            print(f"Embedding similarity: {similarities}")
            
            # Visualize embeddings
            visualize_embeddings(model, config, dataloader, device)
            
            # Log the visualization to wandb
            wandb.log({"embeddings": wandb.Image(os.path.join(config.log_dir, 'tsne_embeddings.png'))})
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config.checkpoint_dir, 'best_model.pt'))
            
            print(f"Saved best model checkpoint with loss: {best_loss:.4f}")
        
        # Save the last model
        if epoch == config.num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config.checkpoint_dir, 'last_model.pt'))
    
    # Close wandb run
    wandb.finish()
    
    # Return the trained model
    return model


def main():
    # Get configuration
    config = get_config()
    
    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device(config['device'])
    
    # Create dataset
    dataset = NoiseRobustSpeechDataset(
        clean_data_path=config['data']['clean_data_path'],
        noise_data_path=config['data']['noise_data_path'],
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        snr_range=config['data']['snr_range']
    )
    
    # Create model
    model = BYOLSpeechModel(config).to(device)
    
    # Train model
    train_model(model, dataset, config, device)
    
if __name__ == "__main__":
    main()