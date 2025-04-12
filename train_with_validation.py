import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import wandb
from transformers import AutoFeatureExtractor

from config.config_utils import get_config
from src.models.byol import BYOLSpeechModel, byol_loss
from src.utils.debugging_utils import check_audio_tensor
from evaluate import evaluate_embedding_similarity, visualize_embeddings
from src.data.dataset import NoiseRobustSpeechDataset

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Training function for one epoch
def train_one_epoch(model: BYOLSpeechModel, 
                    dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    device: torch.device) -> float:
    """
    Train the model for one epoch.
    """
    # Clear GPU cache
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
        
        # Compute BYOL loss
        loss = byol_loss(online_pred, target_proj)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update target network with EMA
        model._update_target_network()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Validation function for one epoch
def validate(model: BYOLSpeechModel, 
             dataloader: DataLoader, 
             device: torch.device) -> float:
    """
    Evaluate the model on a validation dataset.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            
            check_audio_tensor(clean_input_values, "clean_input_values")
            check_audio_tensor(noisy_input_values, "noisy_input_values")
            
            online_pred, target_proj = model(clean_input_values, noisy_input_values)
            check_audio_tensor(online_pred, "online_pred")
            check_audio_tensor(target_proj, "target_proj")
            
            loss = byol_loss(online_pred, target_proj)
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

def main():
    # Get configuration and set seed
    config = get_config()
    set_seed()
    
    # Set device
    device = torch.device(config['device'])
    
    # Initialize wandb run
    wandb.init(project="noise-robust-speech-embeddings", config=config, mode=config['logging'].get('wandb_mode'))
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Create the full dataset using your existing dataset class
    full_dataset = NoiseRobustSpeechDataset(
        clean_data_path = config['data']['clean_data_path'],
        noise_data_path = config['data']['noise_data_path'],
        sample_rate = config['data']['sample_rate'],
        max_audio_length = config['data']['max_audio_length'],
        snr_range = config['data']['snr_range'],
        feature_extractor = feature_extractor
    )
    
    # Split the dataset into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders for training and validation
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create BYOL model and move to device
    model = BYOLSpeechModel(config).to(device)
    
    # Set up optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    total_steps = len(train_dataloader) * config['training']['num_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        # Validation
        val_loss = validate(model, val_dataloader, device)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Evaluate embedding similarity periodically on validation set
        if (epoch + 1) % 5 == 0 or epoch == config['training']['num_epochs'] - 1:
            similarities = evaluate_embedding_similarity(model, val_dataloader, device)
            for snr, sim in similarities.items():
                wandb.log({f"similarity_snr_{snr}": sim})
            print(f"Embedding similarity: {similarities}")
            
        # Visualize embeddings periodically
        if (epoch + 1) % 10 == 0 or epoch == config['training']['num_epochs'] - 1:
            visualize_embeddings(model, val_dataloader, device, config['training']['log_dir'])
            wandb.log({"embeddings": wandb.Image(os.path.join(config['training']['log_dir'], 'tsne_embeddings.png'))})
        
        # Save best model based on validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config['training']['checkpoint_dir'], 'best_model.pt'))
            print(f"Saved best model checkpoint with val loss: {best_val_loss:.4f}")
        
        # Save the final model checkpoint at the end of training
        if epoch == config['training']['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config['training']['checkpoint_dir'], 'last_model.pt'))
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()