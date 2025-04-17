import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionClassifier(nn.Module):
    """
    Emotion classifier for both categorical and dimensional emotion recognition.
    Based on MSP-Podcast Challenge approach.
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
        
        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
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
    
    def forward(self, x, task='both'):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            task: 'categorical', 'dimensional', or 'both'
            
        Returns:
            Tuple of (categorical_logits, dimensional_values) depending on task
        """
        # Encoder
        embeddings = self.encoder(x)
        
        # Shared layers
        shared_features = self.shared_fc(embeddings)
        
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