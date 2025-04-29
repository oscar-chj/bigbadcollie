"""
Visualization utilities for BigBadCollie.

This module provides tools for visualizing chess positions, concepts,
and model predictions to aid in understanding and debugging.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import chess
import chess.svg
from typing import Dict, Optional
import json
import torch

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHESS_CONCEPTS, ALL_CONCEPTS


def plot_training_history(history_path: str):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history_path: Path to the training history JSON file
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.axvline(x=history['best_epoch'] - 1, color='r', linestyle='--', 
                   label=f'Best Model (Epoch {history["best_epoch"]})')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(os.path.dirname(history_path), 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error plotting training history: {e}")


def render_chess_board(board, save_path: Optional[str] = None, show: bool = True):
    """
    Render a chess board as an image.
    
    Args:
        board: A chess.Board object
        save_path: Path to save the rendered image, if provided
        show: Whether to display the image
    """
    try:
        # Generate SVG representation
        svg = chess.svg.board(board, size=350)
        
        # Try to convert to image using cairosvg
        try:
            import cairosvg
            from io import BytesIO
            import matplotlib.image as mpimg
            
            png_data = BytesIO()
            cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_data)
            png_data.seek(0)
            
            img = mpimg.imread(png_data)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            # If cairosvg is not available, save the SVG directly
            if save_path:
                svg_path = save_path.replace('.png', '.svg')
                with open(svg_path, 'w') as f:
                    f.write(svg)
                print(f"Board saved as SVG to {svg_path} (install cairosvg for PNG output)")
            
            print("Warning: cairosvg not available - cannot render board as PNG")
    
    except Exception as e:
        print(f"Error rendering chess board: {e}")


def visualize_concept_distribution(dataset, save_path: Optional[str] = None, show: bool = True):
    """
    Visualize the distribution of chess concepts in a dataset.
    
    Args:
        dataset: A ChessConceptDataset instance
        save_path: Path to save the visualization, if provided
        show: Whether to display the visualization
    """
    try:
        # Count instances of each concept
        concept_counts = np.zeros(len(ALL_CONCEPTS))
        
        for i in range(len(dataset)):
            labels = dataset[i]['labels'].numpy()
            concept_counts += (labels > 0.5).astype(int)
        
        # Group concepts by category
        categories = {}
        for cat, concepts in CHESS_CONCEPTS.items():
            categories[cat] = [ALL_CONCEPTS.index(concept) for concept in concepts]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot all concepts
        x = np.arange(len(ALL_CONCEPTS))
        ax1.bar(x, concept_counts)
        ax1.set_xticks(x)
        ax1.set_xticklabels(ALL_CONCEPTS, rotation=90)
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Chess Concepts in Dataset')
        
        # Plot by category
        cat_data = []
        cat_labels = []
        cat_positions = []
        start = 0
        
        for cat, indices in categories.items():
            cat_counts = [concept_counts[i] for i in indices]
            cat_concepts = [ALL_CONCEPTS[i] for i in indices]
            
            positions = np.arange(start, start + len(indices))
            ax2.bar(positions, cat_counts)
            
            cat_positions.append(start + len(indices) / 2)
            cat_labels.append(cat.capitalize())
            start += len(indices) + 2
        
        # Add category labels
        for i, (pos, label) in enumerate(zip(cat_positions, cat_labels)):
            ax2.text(pos, 5, label, ha='center', fontweight='bold')
            
            if i < len(cat_positions) - 1:
                ax2.axvline(x=(cat_positions[i] + cat_positions[i+1])/2 - 1, 
                           color='gray', linestyle='--', alpha=0.7)
        
        ax2.set_xticks([])
        ax2.set_ylabel('Count')
        ax2.set_title('Chess Concepts by Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        print(f"Error visualizing concept distribution: {e}")


def visualize_model_comparison(models_metrics: Dict[str, Dict], 
                              metric: str = 'f1', 
                              save_path: Optional[str] = None,
                              show: bool = True):
    """
    Compare the performance of different models.
    
    Args:
        models_metrics: Dictionary mapping model names to their evaluation metrics
        metric: Which metric to use for comparison ('precision', 'recall', 'f1', 'accuracy')
        save_path: Path to save the visualization, if provided
        show: Whether to display the visualization
    """
    try:
        # Extract the specified metric for each model
        model_names = list(models_metrics.keys())
        macro_metrics = [models_metrics[model]['macro_avg'][metric] for model in model_names]
        micro_metrics = [models_metrics[model]['micro_avg'][metric] for model in model_names]
        weighted_metrics = [models_metrics[model]['weighted_avg'][metric] for model in model_names]
        
        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar1 = ax.bar(x - width, macro_metrics, width, label='Macro Avg')
        bar2 = ax.bar(x, micro_metrics, width, label='Micro Avg')
        bar3 = ax.bar(x + width, weighted_metrics, width, label='Weighted Avg')
        
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'Model Comparison by {metric.capitalize()} Score')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Display values on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        print(f"Error visualizing model comparison: {e}")


def visualize_feature_maps(model, input_tensor, layer_idx=-1, save_path=None, show=True):
    """
    Visualize feature maps of a convolutional model.
    
    Args:
        model: A PyTorch model (CNN)
        input_tensor: Input tensor for the model
        layer_idx: Index of the layer to visualize
        save_path: Path to save the visualization, if provided
        show: Whether to display the visualization
    """
    try:
        # Check if we have a CNN model
        if not hasattr(model, 'conv_layers'):
            print("Model does not have conv_layers attribute - cannot visualize feature maps")
            return
        
        # Get feature maps
        conv_layers = list(model.conv_layers)
        
        if layer_idx < 0 or layer_idx >= len(conv_layers):
            layer_idx = len(conv_layers) - 1  # Default to last conv layer
        
        # Forward pass to get feature maps
        model.eval()
        with torch.no_grad():
            # Get the selected layer's output
            features = input_tensor
            for i in range(layer_idx + 1):
                features = conv_layers[i](features)
        
        # Move to CPU and convert to numpy
        feature_maps = features.detach().cpu().numpy()
        
        # Only select the first batch item if batch size > 1
        if feature_maps.shape[0] > 1:
            feature_maps = feature_maps[0:1]
        
        # Plot feature maps
        num_features = min(16, feature_maps.shape[1])  # Limit to a reasonable number
        grid_size = int(np.ceil(np.sqrt(num_features)))
        
        plt.figure(figsize=(15, 15))
        
        for i in range(num_features):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(feature_maps[0, i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Filter {i+1}')
        
        plt.suptitle(f'Feature Maps from Layer {layer_idx+1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        print(f"Error visualizing feature maps: {e}")


if __name__ == "__main__":
    print("Run as a module to use visualization utilities")