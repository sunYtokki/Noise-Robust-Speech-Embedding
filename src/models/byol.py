import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BYOLSpeechModel(nn.Module):
    def __init__(self, config):
        """
        BYOL model for speech embeddings.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        # Online network
        self.online_encoder = WavLMEncoder(config.model_name)
        self.online_projector = ProjectionHead(
            self.online_encoder.output_dim,
            config.projection_dim,
            config.projection_dim
        )
        self.online_predictor = PredictionHead(
            config.projection_dim,
            config.prediction_dim,
            config.projection_dim
        )
        
        # Target network (initially a copy of online network)
        self.target_encoder = WavLMEncoder(config.model_name)
        self.target_projector = ProjectionHead(
            self.target_encoder.output_dim,
            config.projection_dim,
            config.projection_dim
        )
        
        # Initialize the target network with the same weights as the online network
        self._copy_weights(self.online_encoder, self.target_encoder)
        self._copy_weights(self.online_projector, self.target_projector)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
        # EMA decay rate
        self.ema_decay = config.ema_decay
        
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target module."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)
    
    def _update_target_network(self):
        """Update target network with EMA of online network."""
        with torch.no_grad():
            for online_param, target_param in zip(self.online_encoder.parameters(), 
                                                self.target_encoder.parameters()):
                target_param.data = self.ema_decay * target_param.data + \
                                  (1 - self.ema_decay) * online_param.data
                
            for online_param, target_param in zip(self.online_projector.parameters(), 
                                                self.target_projector.parameters()):
                target_param.data = self.ema_decay * target_param.data + \
                                  (1 - self.ema_decay) * online_param.data
    
    def forward(self, clean_input_values: torch.Tensor, noisy_input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the BYOL model.
        
        Args:
            clean_input_values: Clean speech input values
            noisy_input_values: Noisy speech input values
            
        Returns:
            online_pred: Online network predictions
            target_proj: Target network projections
        """
        # Online network forward (on clean speech)
        online_emb = self.online_encoder(clean_input_values)
        online_proj = self.online_projector(online_emb)
        online_pred = self.online_predictor(online_proj)
        
        # Target network forward (on noisy speech)
        with torch.no_grad():
            target_emb = self.target_encoder(noisy_input_values)
            target_proj = self.target_projector(target_emb)
            
        return online_pred, target_proj
    
    def get_encoder(self) -> nn.Module:
        """Get the online encoder for downstream tasks."""
        return self.online_encoder


# Loss function
def byol_loss(online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
    """
    BYOL loss function.
    
    Args:
        online_pred: Online network predictions [batch_size, projection_dim]
        target_proj: Target network projections [batch_size, projection_dim]
        
    Returns:
        loss: Mean squared error loss
    """
    # Normalize to unit sphere
    online_pred = F.normalize(online_pred, dim=1)
    target_proj = F.normalize(target_proj, dim=1)
    
    # Mean squared error
    loss = 2 - 2 * (online_pred * target_proj).sum(dim=1).mean()
    
    return loss
