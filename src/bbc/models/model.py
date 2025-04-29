"""
Model architecture module for BigBadCollie.
Implements neural network architectures for chess concept recognition.
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, ALL_CONCEPTS, NNUE_CONFIG

class ConceptRecognitionCNN(nn.Module):
    """
    Convolutional Neural Network for recognizing chess concepts.
    Takes a board representation and outputs concept probabilities.
    """
    
    def __init__(self, config: Dict = MODEL_CONFIG):
        super(ConceptRecognitionCNN, self).__init__()
        self.config = config
        
        # Extract parameters from config
        input_shape = config['input_shape']
        conv_filters = config['conv_filters']
        kernel_size = config['conv_kernel_size']
        fc_layers = config['fc_layers']
        dropout_rate = config['dropout_rate']
        num_concepts = config['num_concepts']
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[2]  # number of piece types
        
        for filters in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size, padding=1),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate)
                )
            )
            in_channels = filters
        
        # Calculate flattened size after convolutions
        # Assuming no stride and padding=1, spatial dimensions remain the same
        flat_size = conv_filters[-1] * input_shape[0] * input_shape[1]
        
        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = flat_size
        
        for features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, features),
                    nn.BatchNorm1d(features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = features
        
        # Output layer - one output per chess concept
        self.output_layer = nn.Linear(in_features, num_concepts)
        
    def forward(self, x):
        # Input shape: (batch_size, 8, 8, 12)
        # Reshape for PyTorch's Conv2d (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply sigmoid activation for multi-label classification
        x = torch.sigmoid(x)
        
        return x


class ConceptRecognitionTransformer(nn.Module):
    """
    Transformer-based model for recognizing chess concepts.
    Takes a board representation and outputs concept probabilities.
    """
    
    def __init__(self, config: Dict = MODEL_CONFIG):
        super(ConceptRecognitionTransformer, self).__init__()
        self.config = config
        
        # Extract parameters
        input_shape = config['input_shape']
        fc_layers = config['fc_layers']
        dropout_rate = config['dropout_rate']
        num_concepts = config['num_concepts']
        
        # Embedding dimension
        self.embed_dim = 128
        
        # Linear projection for each square
        self.position_embedding = nn.Linear(input_shape[2], self.embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=4,
            dim_feedforward=512,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=3
        )
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.embed_dim * input_shape[0] * input_shape[1]  # 8x8 board
        
        for features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, features),
                    nn.LayerNorm(features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = features
        
        # Output layer - one output per chess concept
        self.output_layer = nn.Linear(in_features, num_concepts)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape into sequence of squares: (batch, 64, 12)
        x = x.view(batch_size, -1, self.config['input_shape'][2])
        
        # Project each square into embedding space
        x = self.position_embedding(x)  # (batch, 64, embed_dim)
        
        # Reshape for transformer: (64, batch, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Reshape back: (batch, 64, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply sigmoid for multi-label classification
        x = torch.sigmoid(x)
        
        return x


class NNUEBasedModel(nn.Module):
    """
    Chess concept recognition model based on NNUE architecture.
    This model can either use features from a pretrained NNUE model or
    train its own NNUE-like feature extractor from scratch.
    """
    
    def __init__(self, config: Dict = MODEL_CONFIG, nnue_config: Dict = NNUE_CONFIG):
        super(NNUEBasedModel, self).__init__()
        self.config = config
        self.nnue_config = nnue_config
        
        # Extract parameters
        input_shape = config['input_shape']
        fc_layers = config['fc_layers']
        dropout_rate = config['dropout_rate']
        num_concepts = config['num_concepts']
        
        # NNUE feature transformation
        self.use_halfkp = nnue_config['use_halfkp']
        
        if self.use_halfkp:
            # HalfKP feature size: 64 king positions * 64 squares * 10 piece types (minus king)
            self.halfkp_size = 64 * 64 * 10
            # But we'll use a more manageable subset as a demonstration
            self.halfkp_size = 256
            self.halfkp_embedding = nn.Embedding(self.halfkp_size, 8)
            feature_size = 8 * 2  # 8 dimensions for each color's features
        else:
            # Standard board representation
            feature_size = input_shape[0] * input_shape[1] * input_shape[2]
        
        # Feature transformer layers (similar to NNUE architecture)
        self.feature_layers = nn.ModuleList()
        in_features = feature_size
        
        for i in range(nnue_config['feature_transformer_layers']):
            out_features = 256 if i == 0 else 32
            self.feature_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ClippedReLU(),  # NNUE typically uses clipped ReLU
                )
            )
            in_features = out_features
        
        # Fully connected layers for concept recognition
        self.fc_layers = nn.ModuleList()
        
        for features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            in_features = features
        
        # Output layer - one output per chess concept
        self.output_layer = nn.Linear(in_features, num_concepts)
    
    def _extract_halfkp_features(self, x):
        """
        Extract HalfKP features from board representation.
        This is a simplified implementation for demonstration.
        """
        # In a real implementation, this would transform x into HalfKP indices
        # For now, we'll just create dummy indices
        batch_size = x.size(0)
        white_indices = torch.randint(0, self.halfkp_size, (batch_size, 32), device=x.device)
        black_indices = torch.randint(0, self.halfkp_size, (batch_size, 32), device=x.device)
        
        # Get embeddings
        white_features = self.halfkp_embedding(white_indices).sum(dim=1)
        black_features = self.halfkp_embedding(black_indices).sum(dim=1)
        
        # Concatenate
        features = torch.cat([white_features, black_features], dim=1)
        return features
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features from board representation
        if self.use_halfkp:
            x = self._extract_halfkp_features(x)
        else:
            x = x.view(batch_size, -1)  # Flatten
        
        # Apply feature transformer layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply sigmoid for multi-label classification
        x = torch.sigmoid(x)
        
        return x


def create_model(model_type: str = MODEL_CONFIG['base_model'], 
                 config: Dict = MODEL_CONFIG,
                 nnue_config: Dict = NNUE_CONFIG) -> nn.Module:
    """
    Factory function to create the appropriate model based on configuration.
    """
    if model_type == 'cnn':
        return ConceptRecognitionCNN(config)
    elif model_type == 'transformer':
        return ConceptRecognitionTransformer(config)
    elif model_type == 'nnue_based':
        return NNUEBasedModel(config, nnue_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model_type = MODEL_CONFIG['base_model']
    model = create_model(model_type)
    
    # Create dummy input
    batch_size = 16
    input_shape = MODEL_CONFIG['input_shape']
    x = torch.rand(batch_size, *input_shape)
    
    # Forward pass
    output = model(x)
    
    print(f"Model type: {model_type}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of concepts: {len(ALL_CONCEPTS)}")
    print(f"Output concepts: {ALL_CONCEPTS}")