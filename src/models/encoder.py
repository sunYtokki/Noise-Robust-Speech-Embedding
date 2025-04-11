import torch
import torch.nn as nn
from transformers import AutoModel

class WavLMEncoder(nn.Module):
    def __init__(self, model_name: str):
        """
        WavLM encoder wrapper.
        
        Args:
            model_name: Hugging Face model name
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from WavLM model."""
        # outputs has shape (batch_size, sequence_length, hidden_size)
        outputs = self.model(input_values).last_hidden_state
        
        # Pool the outputs to get a fixed-size representation
        # Using mean pooling over the sequence dimension
        embeddings = torch.mean(outputs, dim=1)
        
        return embeddings
