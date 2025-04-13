import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.models.encoder import WavLMEncoder
from src.models.multi_layer_heads import ProjectionHead, PredictionHead
from src.utils.logging_utils import logger

class BYOLSpeechModel(nn.Module):
    def __init__(self, config):
        """
        BYOL model for speech embeddings.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Get config values
        model_name = config['model']['name']
        projection_dim = config['model']['projection_dim']
        prediction_hidden_dim = config['model']['prediction_dim']
        self.ema_decay = config['model']['ema_decay']
        
        # Online network
        self.online_encoder = WavLMEncoder(model_name)
        self.online_projector = ProjectionHead(
            self.online_encoder.output_dim,
            projection_dim,
            projection_dim
        )
        self.online_predictor = PredictionHead(
            projection_dim,
            prediction_hidden_dim,
            projection_dim
        )
        
        # Target network (initially a copy of online network)
        self.target_encoder = WavLMEncoder(model_name)
        self.target_projector = ProjectionHead(
            self.target_encoder.output_dim,
            projection_dim,
            projection_dim
        )
        
        # Initialize the target network with the same weights as the online network
        self._copy_weights(self.online_encoder, self.target_encoder)
        self._copy_weights(self.online_projector, self.target_projector)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
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


def byol_loss(online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
    """
    BYOL loss function with numerical stability improvements.
    """
    # Check for NaN values before normalization
    if torch.isnan(online_pred).any() or torch.isnan(target_proj).any():
        logger.error("NaN detected in tensors before normalization!")
        
    # Add a small epsilon to prevent all-zero vectors
    online_pred = online_pred + 1e-10
    target_proj = target_proj + 1e-10
    
    # Normalize to unit sphere with a small epsilon
    online_pred = F.normalize(online_pred, dim=1, eps=1e-10)
    target_proj = F.normalize(target_proj, dim=1, eps=1e-10)
    
    # Check for NaN values after normalization
    if torch.isnan(online_pred).any() or torch.isnan(target_proj).any():
        logger.error("NaN detected in tensors after normalization!")
    
    # Compute loss with clamping to avoid extreme values
    similarity = torch.sum(online_pred * target_proj, dim=1)
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)
    loss = 2 - 2 * similarity.mean()
    
    return loss