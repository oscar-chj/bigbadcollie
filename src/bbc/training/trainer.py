"""
Training module for BigBadCollie.
Handles model training, validation, and checkpointing.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
from tqdm import tqdm

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TRAINING_CONFIG, MODEL_CONFIG, ALL_CONCEPTS
from models.model import create_model


class ChessConceptDataset(Dataset):
    """
    Dataset for chess positions and concept labels.
    """
    def __init__(self, data):
        """
        Initialize dataset with processed data.
        
        Args:
            data: List of (board_array, concepts_dict) tuples
        """
        self.board_arrays = []
        self.concept_labels = []
        
        # Extract board arrays and concept labels
        for board_array, concepts_dict in data:
            self.board_arrays.append(board_array)
            
            # Convert concepts dictionary to label array
            label_array = np.zeros(len(ALL_CONCEPTS), dtype=np.float32)
            for i, concept in enumerate(ALL_CONCEPTS):
                if concept in concepts_dict:
                    label_array[i] = concepts_dict[concept]
            
            self.concept_labels.append(label_array)
        
        # Convert to numpy arrays
        self.board_arrays = np.array(self.board_arrays, dtype=np.float32)
        self.concept_labels = np.array(self.concept_labels, dtype=np.float32)
        
    def __len__(self):
        return len(self.board_arrays)
    
    def __getitem__(self, idx):
        return {
            'board': torch.tensor(self.board_arrays[idx], dtype=torch.float32),
            'labels': torch.tensor(self.concept_labels[idx], dtype=torch.float32)
        }


class ConceptTrainer:
    """
    Trainer class for BigBadCollie chess concept recognition model.
    """
    
    def __init__(self, config=TRAINING_CONFIG):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_data_loaders(self, train_data, val_data):
        """Create PyTorch DataLoaders for training and validation."""
        train_dataset = ChessConceptDataset(train_data)
        val_dataset = ChessConceptDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def create_model_and_optimizer(self, model_type=MODEL_CONFIG['base_model']):
        """Create model and optimizer."""
        model = create_model(model_type)
        model = model.to(self.device)
        
        # Create optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        
        return model, optimizer
    
    def binary_cross_entropy_loss(self, outputs, targets):
        """
        Binary cross entropy loss for multi-label classification.
        """
        return nn.BCELoss()(outputs, targets)
    
    def train_epoch(self, model, train_loader, optimizer):
        """Train model for one epoch."""
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            boards = batch['board'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(boards)
            loss = self.binary_cross_entropy_loss(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, model, val_loader):
        """Validate model on validation set."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                boards = batch['board'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(boards)
                loss = self.binary_cross_entropy_loss(outputs, labels)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"model_epoch_{epoch}_valloss_{val_loss:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': MODEL_CONFIG
        }, checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def save_best_model(self, model, optimizer, epoch, val_loss):
        """Save best model."""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            "best_model.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': MODEL_CONFIG
        }, checkpoint_path)
        
        print(f"Saved best model to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train(self, train_data, val_data, model_type=MODEL_CONFIG['base_model']):
        """
        Train model on training data and validate on validation data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            model_type: Type of model to train
            
        Returns:
            Trained model and training history
        """
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_data, val_data)
        
        # Create model and optimizer
        model, optimizer = self.create_model_and_optimizer(model_type)
        
        # Track training progress
        best_epoch = 0
        start_time = time.time()
        
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_loss = self.validate(model, val_loader)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(model, optimizer, epoch, val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_epoch = epoch
                
                # Save best model
                self.save_best_model(model, optimizer, epoch, val_loss)
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                
                # Early stopping
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Training summary
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
        
        history_path = os.path.join(self.config['checkpoint_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists
            history = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in history.items()}
            json.dump(history, f)
        
        return model, history


if __name__ == "__main__":
    # This would be run from the main training script
    pass