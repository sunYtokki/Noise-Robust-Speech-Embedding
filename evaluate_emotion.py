# evaluate_emotion_classifier.py
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from src.models.byol import BYOLSpeechModel
from src.models.emotion_classifier import EmotionClassifier
from data.noisy_speech_dataset import create_dataloaders
from config.config_utils import get_config

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier using BYOL embeddings")
    parser.add_argument("--config", type=str, default="config/emotion_classifier.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--byol_checkpoint", type=str, required=True,
                        help="Path to BYOL checkpoint")
    parser.add_argument("--classifier_checkpoint", type=str, required=True,
                        help="Path to emotion classifier checkpoint")
    parser.add_argument("--output_file", type=str, default="results/evaluation_results.txt",
                        help="File to save evaluation results")
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders (only validation/test set needed)
    _, val_loader = create_dataloaders(config)
    
    # Load BYOL model
    byol_model = BYOLSpeechModel(config)
    byol_checkpoint = torch.load(args.byol_checkpoint, map_location=device)
    byol_model.load_state_dict(byol_checkpoint['model_state_dict'])
    byol_model.to(device)
    
    # Get encoder and set to eval mode
    encoder = byol_model.get_encoder()
    encoder.eval()
    
    # Create emotion classifier
    emotion_classifier = EmotionClassifier(
        input_dim=encoder.output_dim,
        hidden_dim=config['emotion_classifier']['hidden_dim'],
        num_layers=config['emotion_classifier']['num_layers'],
        num_emotions=config['emotion_classifier']['num_emotions'],
        dropout=0.0  # No dropout during evaluation
    )
    
    # Load emotion classifier weights
    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    emotion_classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    emotion_classifier.to(device)
    emotion_classifier.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_noisy_preds = []  # For evaluating on noisy versions
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            clean_input_values = batch["clean_input_values"].to(device)
            noisy_input_values = batch["noisy_input_values"].to(device)
            labels = batch["emotion_labels"].to(device)
            
            # Get embeddings from the BYOL encoder
            clean_embeddings = encoder(clean_input_values)
            noisy_embeddings = encoder(noisy_input_values)
            
            # Forward pass through the emotion classifier
            clean_logits = emotion_classifier(clean_embeddings)
            noisy_logits = emotion_classifier(noisy_embeddings)
            
            # Get predictions
            clean_preds = torch.argmax(clean_logits, dim=1)
            noisy_preds = torch.argmax(noisy_logits, dim=1)
            
            # Save predictions and labels for metrics calculation
            all_preds.append(clean_preds.cpu().numpy())
            all_noisy_preds.append(noisy_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds)
    all_noisy_preds = np.concatenate(all_noisy_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    clean_accuracy = np.mean(all_preds == all_labels)
    noisy_accuracy = np.mean(all_noisy_preds == all_labels)
    
    clean_f1 = f1_score(all_labels, all_preds, average='macro')
    noisy_f1 = f1_score(all_labels, all_noisy_preds, average='macro')
    
    # Generate confusion matrices
    clean_cm = confusion_matrix(all_labels, all_preds)
    noisy_cm = confusion_matrix(all_labels, all_noisy_preds)
    
    # Generate classification reports
    emotion_names = config.get('emotion_names', [f"Emotion_{i}" for i in range(config['emotion_classifier']['num_emotions'])])
    clean_report = classification_report(all_labels, all_preds, target_names=emotion_names)
    noisy_report = classification_report(all_labels, all_noisy_preds, target_names=emotion_names)
    
    # Save results to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write("Evaluation Results for Emotion Classification\n")
        f.write("===========================================\n\n")
        
        f.write("Clean Audio Results:\n")
        f.write(f"Accuracy: {clean_accuracy:.4f}\n")
        f.write(f"F1 Score (Macro): {clean_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(clean_report)
        f.write("\n\n")
        
        f.write("Noisy Audio Results:\n")
        f.write(f"Accuracy: {noisy_accuracy:.4f}\n")
        f.write(f"F1 Score (Macro): {noisy_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(noisy_report)
        f.write("\n\n")
        
        f.write("Improvement (Noisy vs Clean):\n")
        f.write(f"Accuracy Delta: {noisy_accuracy - clean_accuracy:.4f}\n")
        f.write(f"F1 Score Delta: {noisy_f1 - clean_f1:.4f}\n")
    
    print(f"Evaluation complete! Results saved to {args.output_file}")
    print(f"Clean Audio - Accuracy: {clean_accuracy:.4f}, F1 Score: {clean_f1:.4f}")
    print(f"Noisy Audio - Accuracy: {noisy_accuracy:.4f}, F1 Score: {noisy_f1:.4f}")

if __name__ == "__main__":
    main()