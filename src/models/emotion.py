import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pool import AttentiveStatisticsPooling


class EmotionClassifier(nn.Module):
    """
    Emotion classifier for both categorical and dimensional emotion recognition.
    Uses AttentiveStatisticsPooling for improved feature extraction.
    """
    def __init__(self, encoder, hidden_dim=1024, dropout=0.5, num_emotions=8):
        """
        Initialize the emotion classifier.
        
        Args:
            encoder: Pre-trained encoder model (WavLM)
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
            num_emotions: Number of emotion categories
        """
        super(EmotionClassifier, self).__init__()
        self.encoder = encoder
        self.input_dim = encoder.output_dim
        self.hidden_dim = hidden_dim
        
        # Add attentive statistics pooling layer
        self.pooling = AttentiveStatisticsPooling(self.input_dim)
        
        # Double the input dimension as the pooling concatenates mean and std
        pooled_dim = self.input_dim * 2
        
        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Categorical emotion classification layers
        self.categorical_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.categorical_out = nn.Linear(hidden_dim, num_emotions)
        
        # Dimensional emotion regression layers (arousal, valence, dominance)
        self.dimensional_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dimensional_out = nn.Linear(hidden_dim, 3)  # A, V, D
    
    def forward(self, x, attention_mask=None, task='both'):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            attention_mask: Attention mask for variable-length sequences
            task: 'categorical', 'dimensional', or 'both'
            
        Returns:
            Tuple of (categorical_logits, dimensional_values) depending on task
        """
        encoder_outputs = self.encoder(x, attention_mask=attention_mask)

        # Create default mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], encoder_outputs.shape[1], device=x.device)
            
        # Apply pooling
        if hasattr(self, 'pooling'):
            # Apply attentive statistics pooling
            features = self.pooling(encoder_outputs, attention_mask)
        else:
            # Apply mean pooling
            features = torch.mean(encoder_outputs, dim=1)
        
        # Shared layers
        shared_features = self.shared_fc(features)
        
        # Task-specific outputs
        if task == 'categorical' or task == 'both':
            cat_features = self.categorical_fc(shared_features)
            categorical_logits = self.categorical_out(cat_features)
        else:
            categorical_logits = None
            
        if task == 'dimensional' or task == 'both':
            dim_features = self.dimensional_fc(shared_features)
            dimensional_values = self.dimensional_out(dim_features)
        else:
            dimensional_values = None
            
        return categorical_logits, dimensional_values
    
    def freeze_encoder(self):
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def unfreeze_encoder_gradually(self, layers_to_unfreeze):
        """
        Gradually unfreeze encoder layers from top to bottom.
        
        Args:
            layers_to_unfreeze: List of layer indices to unfreeze
        """
        # First freeze all layers
        for name, param in self.encoder.model.named_parameters():
            param.requires_grad = False
            
        # Then unfreeze specified layers
        for layer_idx in layers_to_unfreeze:
            for name, param in self.encoder.model.named_parameters():
                if f"layer.{layer_idx}" in name or f"layers.{layer_idx}" in name:
                    param.requires_grad = True
                    
    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)