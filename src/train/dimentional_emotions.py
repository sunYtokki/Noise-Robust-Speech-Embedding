import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from src.models.byol import BYOLSpeechModel
from src.models.emotion import EmotionClassifier
from src.data.emotion_dataset import EmotionDataset
from src.utils.logging_utils import setup_logger, logger
from src.utils.setup_utils import set_seed


def train_dimensional_emotions(config, device='cuda'):
    """Train model for dimensional emotion recognition (arousal, valence, dominance)."""
    # Set up logging and random seed
    setup_logger(config)
    set_seed(config['training']['seed'])
    
    # Create output directories
    checkpoint_dir = os.path.join(config['emotion']['checkpoint_dir'], 'dimensional')
    log_dir = os.path.join(config['emotion']['log_dir'], 'dimensional')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="nrse-emotion-dimensional",
        config=config,
        mode=config['logging'].get('wandb_mode', 'disabled')
    )
    
    # Load feature extractor
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Load pre-trained encoder
    logger.info(f"Loading pre-trained encoder from {config['emotion']['encoder_checkpoint']}")
    byol_model = BYOLSpeechModel(config)
    checkpoint = torch.load(config['emotion']['encoder_checkpoint'], map_location=device)
    byol_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = byol_model.get_encoder()
    
    # Create emotion classifier
    logger.info("Creating emotion classifier for dimensional prediction")
    model = EmotionClassifier(
        encoder=encoder,
        hidden_dim=config['emotion']['hidden_dim'],
        dropout=config['emotion']['dropout_rate'],
        num_emotions=len(EmotionDataset.VALID_EMOTIONS_MAP)  # Not used for dimensional task
    ).to(device)
    
    # Initially freeze encoder
    model.freeze_encoder()
    logger.info(f"Trainable parameters after freezing encoder: {model.get_trainable_params()}")
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = EmotionDataset(
        labels_file=config['emotion']['labels_file'],
        audio_dir=config['emotion']['audio_dir'],
        split="Train",
        feature_extractor=feature_extractor,
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        add_noise=config['emotion']['add_noise_during_training'],
        noise_dir=config['data']['noise_data_path'],
        snr_range=config['data']['snr_range'],
        emotion_mapping=EmotionDataset.VALID_EMOTIONS_MAP,
        categorical_only=False  # Include all samples for dimensional task
    )
    
    val_dataset = EmotionDataset(
        labels_file=config['emotion']['labels_file'],
        audio_dir=config['emotion']['audio_dir'],
        split="Development",
        feature_extractor=feature_extractor,
        sample_rate=config['data']['sample_rate'],
        max_audio_length=config['data']['max_audio_length'],
        emotion_mapping=EmotionDataset.VALID_EMOTIONS_MAP,
        categorical_only=False  # Include all samples for dimensional task
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['emotion']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['emotion']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['emotion']['learning_rate'],
        weight_decay=config['emotion']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config['emotion']['scheduler_patience'],
        verbose=True
    )
    
    # Training loop
    logger.info("Starting dimensional emotion training")
    best_val_ccc = 0.0
    patience_counter = 0
    
    for epoch in range(config['emotion']['classifier_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['emotion']['classifier_epochs']}")
        
        # Train for one epoch
        train_loss, train_ccc = train_one_epoch_dimensional(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_loss, val_ccc = validate_dimensional(
            model, val_loader, device, log_dir
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Avg CCC: {train_ccc['avg']:.4f}")
        logger.info(f"Train CCC - Arousal: {train_ccc['A']:.4f}, Valence: {train_ccc['V']:.4f}, Dominance: {train_ccc['D']:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Avg CCC: {val_ccc['avg']:.4f}")
        logger.info(f"Val CCC - Arousal: {val_ccc['A']:.4f}, Valence: {val_ccc['V']:.4f}, Dominance: {val_ccc['D']:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ccc_avg": train_ccc['avg'],
            "train_ccc_arousal": train_ccc['A'],
            "train_ccc_valence": train_ccc['V'],
            "train_ccc_dominance": train_ccc['D'],
            "val_loss": val_loss,
            "val_ccc_avg": val_ccc['avg'],
            "val_ccc_arousal": val_ccc['A'],
            "val_ccc_valence": val_ccc['V'],
            "val_ccc_dominance": val_ccc['D'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Update scheduler
        scheduler.step(val_ccc['avg'])
        
        # Save best model
        if val_ccc['avg'] > best_val_ccc:
            best_val_ccc = val_ccc['avg']
            logger.info(f"New best validation CCC: {best_val_ccc:.4f}")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ccc': val_ccc,
                'val_loss': val_loss,
                'train_ccc': train_ccc,
                'train_loss': train_loss,
                'config': config
            }, os.path.join(checkpoint_dir, 'best_classifier_model.pt'))
            
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{config['emotion']['patience']}")
        
            # Early stopping
        if patience_counter >= config['emotion']['patience']:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Fine-tuning with encoder unfreezing
    if config['emotion']['unfreeze_encoder']:
        logger.info("Starting fine-tuning with encoder unfreezing")
        
        # Load best classifier model
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_classifier_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Reset best CCC and patience counter
        best_val_ccc = checkpoint['val_ccc']['avg']
        patience_counter = 0
        
        # Create new optimizer with lower learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['emotion']['fine_tuning_lr'],
            weight_decay=config['emotion']['weight_decay']
        )
        
        # Create new scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=config['emotion']['scheduler_patience'],
            verbose=True
        )
        
        # Gradually unfreeze layers
        total_layers = 24  # WavLM base has 24 layers
        
        for epoch in range(config['emotion']['fine_tuning_epochs']):
            logger.info(f"Fine-tuning Epoch {epoch+1}/{config['emotion']['fine_tuning_epochs']}")
            
            # Unfreeze some layers based on schedule
            if epoch < config['emotion']['fine_tuning_epochs']:
                # Calculate layers to unfreeze
                unfreeze_ratio = (epoch + 1) / config['emotion']['fine_tuning_epochs']
                layers_to_unfreeze = list(range(
                    int(total_layers * (1 - unfreeze_ratio)),
                    total_layers
                ))
                
                # Unfreeze layers
                model.unfreeze_encoder_gradually(layers_to_unfreeze)
                logger.info(f"Unfreezing layers {layers_to_unfreeze}")
                logger.info(f"Trainable parameters: {model.get_trainable_params()}")
            
            # Train and validate
            train_loss, train_ccc = train_one_epoch_dimensional(
                model, train_loader, optimizer, device
            )
            
            val_loss, val_ccc = validate_dimensional(
                model, val_loader, device, log_dir
            )
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Avg CCC: {train_ccc['avg']:.4f}")
            logger.info(f"Train CCC - Arousal: {train_ccc['A']:.4f}, Valence: {train_ccc['V']:.4f}, Dominance: {train_ccc['D']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Avg CCC: {val_ccc['avg']:.4f}")
            logger.info(f"Val CCC - Arousal: {val_ccc['A']:.4f}, Valence: {val_ccc['V']:.4f}, Dominance: {val_ccc['D']:.4f}")
            
            wandb.log({
                "fine_tuning_epoch": epoch,
                "train_loss": train_loss,
                "train_ccc_avg": train_ccc['avg'],
                "train_ccc_arousal": train_ccc['A'],
                "train_ccc_valence": train_ccc['V'],
                "train_ccc_dominance": train_ccc['D'],
                "val_loss": val_loss,
                "val_ccc_avg": val_ccc['avg'],
                "val_ccc_arousal": val_ccc['A'],
                "val_ccc_valence": val_ccc['V'],
                "val_ccc_dominance": val_ccc['D'],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "unfrozen_layers": len(layers_to_unfreeze)
            })
            
            # Update scheduler
            scheduler.step(val_ccc['avg'])
            
            # Save best model
            if val_ccc['avg'] > best_val_ccc:
                best_val_ccc = val_ccc['avg']
                logger.info(f"New best validation CCC: {best_val_ccc:.4f}")
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ccc': val_ccc,
                    'val_loss': val_loss,
                    'train_ccc': train_ccc,
                    'train_loss': train_loss,
                    'config': config
                }, os.path.join(checkpoint_dir, 'best_fine_tuned_model.pt'))
                
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Patience counter: {patience_counter}/{config['emotion']['patience']}")
            
            # Early stopping
            if patience_counter >= config['emotion']['patience']:
                logger.info(f"Early stopping triggered after fine-tuning epoch {epoch+1}")
                break
    
    # Finish wandb run
    wandb.finish()
    logger.info(f"Dimensional emotion training complete! Best CCC: {best_val_ccc:.4f}")
    
    return best_val_ccc


def train_one_epoch_dimensional(model, dataloader, optimizer, device):
    """Train the model for one epoch on dimensional emotion regression task."""
    model.train()
    epoch_loss = 0.0
    all_preds = {'A': [], 'V': [], 'D': []}
    all_labels = {'A': [], 'V': [], 'D': []}
    
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch["input_values"].to(device)
        arousal = batch["A"].to(device)
        valence = batch["V"].to(device)
        dominance = batch["D"].to(device)
        
        # Get attention mask if available, otherwise create a default one
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(device)
        else:
            attention_mask = torch.ones(inputs.size(0), inputs.size(1)).to(device)
        
        # Combine dimensions for model input
        labels = torch.stack([arousal, valence, dominance], dim=1)
        
        # Forward pass with attention mask
        _, dimensional_values = model(inputs, attention_mask=attention_mask, task='dimensional')
        
        # Calculate CCC loss
        loss = ccc_loss(dimensional_values, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        
        # Store predictions and labels for CCC calculation
        all_preds['A'].extend(dimensional_values[:, 0].detach().cpu().numpy())
        all_preds['V'].extend(dimensional_values[:, 1].detach().cpu().numpy())
        all_preds['D'].extend(dimensional_values[:, 2].detach().cpu().numpy())
        all_labels['A'].extend(arousal.cpu().numpy())
        all_labels['V'].extend(valence.cpu().numpy())
        all_labels['D'].extend(dominance.cpu().numpy())
    
    # Compute CCC for each dimension
    ccc_values = {}
    ccc_values['A'] = compute_ccc(np.array(all_preds['A']), np.array(all_labels['A']))
    ccc_values['V'] = compute_ccc(np.array(all_preds['V']), np.array(all_labels['V']))
    ccc_values['D'] = compute_ccc(np.array(all_preds['D']), np.array(all_labels['D']))
    ccc_values['avg'] = (ccc_values['A'] + ccc_values['V'] + ccc_values['D']) / 3
    
    return epoch_loss / len(dataloader), ccc_values


def validate_dimensional(model, dataloader, device, log_dir=None):
    """Validate the model on dimensional emotion regression task."""
    model.eval()
    val_loss = 0.0
    all_preds = {'A': [], 'V': [], 'D': []}
    all_labels = {'A': [], 'V': [], 'D': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = batch["input_values"].to(device)
            arousal = batch["A"].to(device)
            valence = batch["V"].to(device)
            dominance = batch["D"].to(device)
            
            # Get attention mask if available, otherwise create a default one
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = torch.ones(inputs.size(0), inputs.size(1)).to(device)
            
            # Combine dimensions for model input
            labels = torch.stack([arousal, valence, dominance], dim=1)
            
            # Forward pass with attention mask
            _, dimensional_values = model(inputs, attention_mask=attention_mask, task='dimensional')
            
            # Calculate CCC loss
            loss = ccc_loss(dimensional_values, labels)
            
            # Track metrics
            val_loss += loss.item()
            
            # Store predictions and labels for CCC calculation
            all_preds['A'].extend(dimensional_values[:, 0].detach().cpu().numpy())
            all_preds['V'].extend(dimensional_values[:, 1].detach().cpu().numpy())
            all_preds['D'].extend(dimensional_values[:, 2].detach().cpu().numpy())
            all_labels['A'].extend(arousal.cpu().numpy())
            all_labels['V'].extend(valence.cpu().numpy())
            all_labels['D'].extend(dominance.cpu().numpy())
    
    # Compute CCC for each dimension
    ccc_values = {}
    ccc_values['A'] = compute_ccc(np.array(all_preds['A']), np.array(all_labels['A']))
    ccc_values['V'] = compute_ccc(np.array(all_preds['V']), np.array(all_labels['V']))
    ccc_values['D'] = compute_ccc(np.array(all_preds['D']), np.array(all_labels['D']))
    ccc_values['avg'] = (ccc_values['A'] + ccc_values['V'] + ccc_values['D']) / 3
    
    # Plot scatter plots for each dimension if log_dir provided
    if log_dir:
        for dim in ['A', 'V', 'D']:
            dimension_name = {'A': 'Arousal', 'V': 'Valence', 'D': 'Dominance'}[dim]
            plot_scatter(
                all_preds[dim], 
                all_labels[dim], 
                os.path.join(log_dir, f'val_scatter_{dim}.png'),
                title=f'Validation {dimension_name} (CCC={ccc_values[dim]:.4f})',
                x_label='Predicted',
                y_label='Ground Truth'
            )
            
            if wandb.run is not None:
                wandb.log({f"val_scatter_{dim}": wandb.Image(os.path.join(log_dir, f'val_scatter_{dim}.png'))})
    
    return val_loss / len(dataloader), ccc_values


def ccc_loss(predictions, targets):
    """Concordance Correlation Coefficient loss function."""
    # Calculate CCC for each dimension and average
    batch_size = predictions.size(0)
    ccc_loss = 0.0
    
    if batch_size > 1:
        for i in range(predictions.size(1)):  # For each dimension (A, V, D)
            pred = predictions[:, i]
            target = targets[:, i]
            
            mean_pred = torch.mean(pred)
            mean_target = torch.mean(target)
            
            var_pred = torch.var(pred, unbiased=False)
            var_target = torch.var(target, unbiased=False)
            
            covar = torch.mean((pred - mean_pred) * (target - mean_target))
            
            ccc = 2 * covar / (var_pred + var_target + (mean_pred - mean_target) ** 2 + 1e-10)
            ccc_loss += 1 - ccc
    
    # Return average CCC loss across dimensions
    return ccc_loss / predictions.size(1)


def compute_ccc(predictions, targets):
    """Compute Concordance Correlation Coefficient."""
    mean_pred = np.mean(predictions)
    mean_target = np.mean(targets)
    
    var_pred = np.var(predictions)
    var_target = np.var(targets)
    
    covar = np.mean((predictions - mean_pred) * (targets - mean_target))
    
    ccc = 2 * covar / (var_pred + var_target + (mean_pred - mean_target) ** 2 + 1e-10)
    
    return ccc


def plot_scatter(predictions, targets, filename, title='Scatter Plot', x_label='Predicted', y_label='Ground Truth'):
    """Plot and save scatter plot of predictions vs targets."""
    plt.figure(figsize=(8, 8))
    plt.scatter(predictions, targets, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(predictions), min(targets))
    max_val = max(max(predictions), max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()