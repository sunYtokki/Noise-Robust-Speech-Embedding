import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import wandb
from transformers import AutoFeatureExtractor

from config.config_utils import get_config
from src.models.byol import BYOLSpeechModel, byol_loss
from src.data.dataset import create_dataloader
from src.utils.debugging_utils import check_audio_tensor
from evaluate import evaluate_embedding_similarity, visualize_embeddings

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
    # Force completion of all GPU operations and free memory
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model.train()
    epoch_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move inputs to device
        clean_input_values = batch["clean_input_values"].to(device)
        noisy_input_values = batch["noisy_input_values"].to(device)

        check_audio_tensor(clean_input_values, "clean_input_values")
        check_audio_tensor(noisy_input_values, "noisy_input_values")
        
        # Forward pass
        online_pred, target_proj = model(clean_input_values, noisy_input_values)

        check_audio_tensor(online_pred, "online_pred")
        check_audio_tensor(target_proj, "target_proj")
        
        # Compute loss
        loss = byol_loss(online_pred, target_proj)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        
        # Update target network with EMA
        model._update_target_network()
        
        # Update learning rate
        scheduler.step()
        
        # Track loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def main():
    # Get configuration
    config = get_config()
    
    # Set seed for reproducibility
    set_seed()
    
    # Set device
    device = torch.device(config['device'])
    
    # Initialize wandb
    wandb.init(project="noise-robust-speech-embeddings", config=config, mode=config['logging'].get('wandb_mode'))
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Create dataloader
    dataloader = create_dataloader(config, feature_extractor)
    
    # Create model
    model = BYOLSpeechModel(config).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(dataloader) * config['training']['num_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Loss: {epoch_loss:.4f}")
        
        # Evaluate embedding similarity
        if (epoch + 1) % config['logging']['metric_logging_interval'] == 0 or epoch == config['training']['num_epochs'] - 1:
            similarities = evaluate_embedding_similarity(model, dataloader, device)
            
            for snr, sim in similarities.items():
                wandb.log({f"similarity_snr_{snr}": sim})
            
            print(f"Embedding similarity: {similarities}")
            
        # Visualize embeddings
        if (epoch + 1) % config['logging']['visualization_interval'] == 0 or epoch == config['training']['num_epochs'] - 1:
            visualize_embeddings(model, dataloader, device, config['training']['log_dir'], epoch + 1)
            
            # Log the visualization to wandb
            wandb.log({"embeddings": wandb.Image(os.path.join(config['training']['log_dir'], 'tsne_embeddings.png'))})
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config['training']['checkpoint_dir'], 'best_model.pt'))
            
            print(f"Saved best model checkpoint with loss: {best_loss:.4f}")
        
        # Save the last model
        if epoch == config['training']['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(config['training']['checkpoint_dir'], 'last_model.pt'))
    
    # Close wandb run
    wandb.finish()
    
    print("Training complete!")

if __name__ == "__main__":
    main()