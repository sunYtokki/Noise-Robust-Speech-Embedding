# train.py (modified version)

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from transformers import AutoFeatureExtractor

from config.config_utils import get_config
from src.models.byol import BYOLSpeechModel, byol_loss
from src.data.dataset import create_dataloaders
from src.utils.debugging_utils import check_audio_tensor
from evaluate import visualize_embeddings, validate_model
from src.utils.logging_utils import setup_logger

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
                    device: torch.device, 
                    config) -> float:
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

        check_audio_tensor(clean_input_values, "clean_input_values", config)
        check_audio_tensor(noisy_input_values, "noisy_input_values", config)
        
        # Forward pass
        online_pred, target_proj = model(clean_input_values, noisy_input_values)

        check_audio_tensor(online_pred, "online_pred", config)
        check_audio_tensor(target_proj, "target_proj", config)
        
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


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improvement = self.best_score - score > self.min_delta
        else:  # mode == 'max'
            improvement = score - self.best_score > self.min_delta
            
        if improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def main():
    # Get configuration
    config = get_config()
    
    # Initial set up
    setup_logger(config)
    set_seed(config['training'].get('seed', 42))
    device = torch.device(config['device'])
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="noise-robust-speech-embeddings", config=config, mode=config['logging'].get('wandb_mode'))
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, feature_extractor)
    
    # Create model
    model = BYOLSpeechModel(config).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Initialize early stopping
    early_stopping_metric = config['training'].get('early_stopping_metric', 'val_loss')
    early_stopping_mode = 'min' if early_stopping_metric == 'val_loss' else 'max'
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 5),
        mode=early_stopping_mode
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_similarity = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config)
        
        # Validate the model
        val_loss, val_metrics = validate_model(model, val_loader, device, config)
        
        # Log metrics
        log_dict = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_avg_similarity": val_metrics['val_avg_similarity'],
            "learning_rate": scheduler.get_last_lr()[0]
        }
        
        # Add SNR-specific similarity metrics
        for snr, sim in val_metrics['val_similarities'].items():
            log_dict[f"val_similarity_snr_{snr}"] = sim
        
        wandb.log(log_dict)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Avg Similarity: {val_metrics['val_avg_similarity']:.4f}")
        
        # Check early stopping based on configured metric
        early_stopping_value = val_loss if early_stopping_metric == 'val_loss' else val_metrics['val_avg_similarity']
        if early_stopping(early_stopping_value):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Visualize embeddings periodically
        if (epoch + 1) % config['logging']['visualization_interval'] == 0 or epoch == config['training']['num_epochs'] - 1:
            img_file_name = f"tsne_embeddings_epoch_{epoch+1}.png"
            visualize_embeddings(model, val_loader, device, config['training']['log_dir'], img_file_name)
            
            # Log the visualization to wandb
            wandb.log({"embeddings": wandb.Image(os.path.join(config['training']['log_dir'], img_file_name))})
        
        # Save checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'similarity': val_metrics['val_avg_similarity'],
            }, os.path.join(config['training']['checkpoint_dir'], 'best_val_loss.pt'))
            
            print(f"Saved best model checkpoint with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint based on validation similarity
        if val_metrics['val_avg_similarity'] > best_val_similarity:
            best_val_similarity = val_metrics['val_avg_similarity']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'similarity': val_metrics['val_avg_similarity'],
            }, os.path.join(config['training']['checkpoint_dir'], 'best_val_similarity.pt'))
            
            print(f"Saved best model checkpoint with validation similarity: {best_val_similarity:.4f}")
        
        # Save the last model
        if epoch == config['training']['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'similarity': val_metrics['val_avg_similarity'],
            }, os.path.join(config['training']['checkpoint_dir'], 'last_model.pt'))
    
    # Close wandb run
    wandb.finish()
    
    print("Training complete!")

if __name__ == "__main__":
    main()