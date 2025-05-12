import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from collections import Counter

from src.models.byol import BYOLSpeechModel
from src.models.emotion import EmotionClassifier
from src.data.emotion_dataset import EmotionDataset, create_emotion_dataloaders
from src.utils.logging_utils import setup_logger, logger
from src.utils.setup_utils import set_seed

def train_categorical_emotions(config, device='cuda'):
    """Train model for categorical emotion recognition."""
    # Set up logging and random seed
    setup_logger(config)
    set_seed(config['training']['seed'])
    
    # Create output directories
    checkpoint_dir = os.path.join(config['emotion']['checkpoint_dir'], 'categorical')
    log_dir = os.path.join(config['training']['log_dir'], 'categorical')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="nrse-emotion-categorical",
        config=config,
        mode=config['logging'].get('wandb_mode', 'disabled')
    )
    
    # Load feature extractor
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(config['model']['name'])
    
    # Load pre-trained encoder
    byol_model = BYOLSpeechModel(config)
    if 'encoder_checkpoint' in config['emotion'] and config['emotion']['encoder_checkpoint']:
        logger.info(f"Loading pre-trained encoder from {config['emotion']['encoder_checkpoint']}")
        checkpoint = torch.load(config['emotion']['encoder_checkpoint'], map_location=device)
        byol_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = byol_model.get_encoder()
    
    # Get data loaders
    train_loader, val_loader = create_emotion_dataloaders(config, feature_extractor)

    # Number of emotion classes
    num_classes = len(train_loader.dataset.emotion_mapping)
    
    # Create emotion classifier
    logger.info(f"Creating emotion classifier with {num_classes} classes")
    model = EmotionClassifier(
        encoder=encoder,
        hidden_dim=config['emotion']['hidden_dim'],
        dropout=config['emotion']['dropout_rate'],
        num_emotions=num_classes
    ).to(device)
    
    # Initially freeze encoder
    model.freeze_encoder()
    logger.info(f"Trainable parameters after freezing encoder: {model.get_trainable_params()}")
    
    # Create datasets
    logger.info("Creating datasets")

    
    # Compute class weights for balanced training
    if config['emotion']['use_class_weights']:
        logger.info("Computing class weights")
        train_labels = Counter([sample['category_idx'] for sample in train_loader.dataset.samples])
        class_weights = compute_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(device)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None
    
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
    logger.info("Starting categorical emotion training")
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Get emotion names
    emotion_names = train_loader.dataset.idx_to_emotion
    
    for epoch in range(config['emotion']['classifier_epochs']):
        print(f"Epoch {epoch+1}/{config['emotion']['classifier_epochs']}")
        
        # Train for one epoch
        train_loss, train_f1 = train_one_epoch_categorical(
            model, train_loader, optimizer, device, class_weights, num_classes
        )
        
        # Validate
        val_loss, val_f1, cm, report = validate_categorical(
            model, val_loader, device, class_weights, emotion_names, log_dir, num_classes
        )
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Log per-class metrics to wandb
        if report:
            for cls_idx, metrics in report.items():
                if isinstance(cls_idx, str) and cls_idx in emotion_names.values():
                    wandb.log({
                        f"val_f1_{cls_idx}": metrics['f1-score'],
                        f"val_precision_{cls_idx}": metrics['precision'],
                        f"val_recall_{cls_idx}": metrics['recall']
                    })
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"New best validation F1: {best_val_f1:.4f}")
            print(f"New best validation F1: {best_val_f1:.4f}")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'train_f1': train_f1,
                'train_loss': train_loss,
                'config': config
            }, os.path.join(checkpoint_dir, 'best_classifier_model.pt'))
            
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{config['emotion']['patience']}")

        
        # Early stopping
        if patience_counter >= config['emotion']['patience']:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Fine-tuning with encoder unfreezing
    if config['emotion']['unfreeze_encoder']:
        logger.info("Starting fine-tuning with encoder unfreezing")
        print("Starting fine-tuning with encoder unfreezing")
        
        # Load best classifier model
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_classifier_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Reset best F1 and patience counter
        best_val_f1 = checkpoint['val_f1']
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
        
        # TODO: match with the other training loop
        for epoch in range(config['emotion']['fine_tuning_epochs']):
            logger.info(f"Fine-tuning Epoch {epoch+1}/{config['emotion']['fine_tuning_epochs']}")
            print(f"Fine-tuning Epoch {epoch+1}/{config['emotion']['fine_tuning_epochs']}")
            
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
            train_loss, train_f1 = train_one_epoch_categorical(
                model, train_loader, optimizer, device, class_weights, num_classes
            )
            
            val_loss, val_f1, cm, report = validate_categorical(
                model, val_loader, device, class_weights, emotion_names, log_dir, num_classes
            )
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            wandb.log({
                "fine_tuning_epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "unfrozen_layers": len(layers_to_unfreeze)
            })
            
            # Log per-class metrics to wandb
            if report:
                for cls_idx, metrics in report.items():
                    if isinstance(cls_idx, str) and cls_idx in emotion_names.values():
                        wandb.log({
                            f"ft_val_f1_{cls_idx}": metrics['f1-score'],
                            f"ft_val_precision_{cls_idx}": metrics['precision'],
                            f"ft_val_recall_{cls_idx}": metrics['recall']
                        })
            
            # Update scheduler
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                logger.info(f"New best validation F1: {best_val_f1:.4f}")
                print(f"New best validation F1: {best_val_f1:.4f}")
                
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                    'train_f1': train_f1,
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
                print(f"Early stopping triggered after fine-tuning epoch {epoch+1}")
                break
    
    # Finish wandb run
    wandb.finish()
    logger.info(f"Categorical emotion training complete! Best F1: {best_val_f1:.4f}")
    print(f"Categorical emotion training complete! Best F1: {best_val_f1:.4f}")
    
    return best_val_f1


def train_one_epoch_categorical(model, dataloader, optimizer, device, class_weights=None, num_classes=8):
    """Train the model for one epoch on categorical emotion classification task."""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch["input_values"].to(device)
        labels = batch["C"].to(device)
        
        # Get attention mask if available, otherwise create a default one
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(device)
        else:
            attention_mask = torch.ones(inputs.size(0), inputs.size(1)).to(device)
        
        # Forward pass with attention mask
        categorical_logits, _ = model(inputs, attention_mask=attention_mask, task='categorical')
        
        # Custom loss function that handles invalid labels
        loss = CE_weight_category(categorical_logits, labels, class_weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        preds = torch.argmax(categorical_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics - filter invalid labels for F1 calculation
    valid_indices = np.array(all_labels) >= 0
    filtered_preds = np.array(all_preds)[valid_indices]
    filtered_labels = np.array(all_labels)[valid_indices]
    
    if len(filtered_labels) > 0:
        f1 = f1_score(filtered_labels, filtered_preds, average='macro', labels=range(num_classes))
    else:
        f1 = 0.0
    
    return epoch_loss / len(dataloader), f1


def validate_categorical(model, dataloader, device, class_weights=None, emotion_names=None, log_dir=None, num_classes=8):
    """Validate the model on categorical emotion classification task."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = batch["input_values"].to(device)
            labels = batch["C"].to(device)
            
            # Get attention mask if available
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = torch.ones(inputs.size(0), inputs.size(1)).to(device)
            
            # Forward pass with attention mask
            categorical_logits, _ = model(inputs, attention_mask=attention_mask, task='categorical')
            
            # Use the same custom loss function as in training
            loss = CE_weight_category(categorical_logits, labels, class_weights)
            
            # Track metrics
            val_loss += loss.item()
            preds = torch.argmax(categorical_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Filter out invalid labels for metrics computation
    valid_indices = np.array(all_labels) >= 0
    filtered_preds = np.array(all_preds)[valid_indices]
    filtered_labels = np.array(all_labels)[valid_indices]
    
    # Check if there are any valid labels
    if len(filtered_labels) == 0:
        logger.warning("No valid labels for validation!")
        return val_loss / len(dataloader), 0.0, None, None
    
    # Compute metrics
    f1 = f1_score(filtered_labels, filtered_preds, average='macro', labels=range(num_classes))
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=range(num_classes))
    
    # Generate classification report
    if emotion_names:
        report = classification_report(
            filtered_labels, 
            filtered_preds, 
            labels=range(num_classes),
            target_names=[emotion_names.get(i, f"Class {i}") for i in range(num_classes)],
            output_dict=True
        )
        
        # Log per-class metrics
        logger.info("Per-class validation metrics:")
        for cls_idx in range(num_classes):
            cls_name = emotion_names.get(cls_idx, f"Class {cls_idx}")
            if cls_name in report:
                logger.info(f"  {cls_name}: F1={report[cls_name]['f1-score']:.4f}, Precision={report[cls_name]['precision']:.4f}, Recall={report[cls_name]['recall']:.4f}")
    else:
        report = None
    
    # Plot confusion matrix if emotion names provided
    if emotion_names and log_dir:
        plot_confusion_matrix(
            cm, 
            [emotion_names.get(i, f"Class {i}") for i in range(num_classes)], 
            filename=os.path.join(log_dir, 'val_confusion_matrix.png'),
            title='Validation Confusion Matrix'
        )
        
        if wandb.run is not None:
            wandb.log({"val_confusion_matrix": wandb.Image(os.path.join(log_dir, 'val_confusion_matrix.png'))})
    
    return val_loss / len(dataloader), f1, cm, report


def compute_class_weights(labels, num_classes):
    """Compute class weights based on label distribution."""
    # count = Counter(labels)
    class_weights = torch.ones(num_classes)
    total = sum(labels.values())
    
    for cls, cnt in labels.items():
        if cls >= 0 and cls < num_classes:  # Ensure valid class index
            # Apply stronger weight to rare classes
            class_weights[cls] = (total / (cnt * num_classes)) ** 1.5  # Exponent for stronger weighting
    
    return class_weights

# TODO: try focal loss
def focal_loss(logits, labels, gamma=2.0, alpha=None):
    """Focal loss for multi-class classification with imbalanced data."""
    ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=alpha)
    pt = torch.exp(-ce_loss)
    loss = (1 - pt) ** gamma * ce_loss
    return loss.mean()

def CE_weight_category(logits, labels, weights=None):
    """
    Custom cross entropy loss that handles invalid labels without using ignore_index.
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        labels: Target labels [batch_size]
        weights: Optional class weights [num_classes]
        
    Returns:
        loss: Scalar loss value
    """
    # Create mask for valid labels (non-negative and within range)
    valid_mask = (labels >= 0) & (labels < logits.size(1))
    
    # Check if any valid samples exist
    if not valid_mask.any():
        # Return zero loss if no valid samples (with grad tracking)
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Filter to valid samples only
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Apply standard cross entropy with optional class weights
    if weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    return loss_fn(valid_logits, valid_labels)


def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png', title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()