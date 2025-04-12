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
        # Check input shape and reshape if needed
        if input_values.dim() == 3:  # [batch_size, 1, sequence_length]
            input_values = input_values.squeeze(1)  # [batch_size, sequence_length]
        
        # outputs has shape (batch_size, sequence_length, hidden_size)
        outputs = self.model(input_values).last_hidden_state
        
        # Pool the outputs to get a fixed-size representation
        # Using mean pooling over the sequence dimension
        embeddings = torch.mean(outputs, dim=1)
        
        return embeddings

def main():
    # test the WavLMEncoder
    model_name = "microsoft/wavlm-base-plus"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = WavLMEncoder(model_name)
    encoder.to(device)
    encoder.eval()
    
    # Create dummy input:
    # For instance, simulate a batch of 2 samples with 1-channel audio of 16000 samples (e.g., 1 second at 16 kHz)
    batch_size = 2
    sequence_length = 16000  # adjust this length based on your requirements
    dummy_input = torch.randn(batch_size, 1, sequence_length, device=device)
    
    # Forward pass through the encoder without gradient computation
    with torch.no_grad():
        embeddings = encoder(dummy_input)
    
    # Print the resulting embeddings shape and check for abnormal values
    print("Embeddings shape:", embeddings.shape)

    if torch.isnan(embeddings).any():
        print("Warning: Embeddings contain NaN values!")
    else:
        print("Embeddings are numerically stable.")

    if torch.isinf(embeddings).any():
        print("Warning: Embeddings contain Inf values!")
    else:
        print("No infinite values in embeddings.")
    
if __name__ == "__main__":
    main()