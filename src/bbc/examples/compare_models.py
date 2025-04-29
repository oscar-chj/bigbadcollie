#!/usr/bin/env python3
"""
Script to train and compare different model architectures in BigBadCollie.
"""
import os
import sys
import time
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BigBadCollie modules
from config.config import DATA_CONFIG, TRAINING_CONFIG, MODEL_CONFIG
from data.processor import ChessDataProcessor
from training.trainer import ConceptTrainer
from evaluation.evaluator import ConceptEvaluator
from utils.visualize import plot_training_history, visualize_model_comparison
from torch.utils.data import DataLoader
from training.trainer import ChessConceptDataset


def train_and_compare_models(data_dir=None, output_dir=None, model_types=None, epochs=10):
    """
    Train and compare different model architectures.
    
    Args:
        data_dir: Directory with processed data
        output_dir: Directory to save results
        model_types: List of model types to compare (e.g., ['cnn', 'transformer', 'nnue_based'])
        epochs: Number of training epochs
    """
    if model_types is None:
        model_types = ['cnn', 'transformer', 'nnue_based']
        
    if output_dir is None:
        output_dir = 'model_comparison_results'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    # Use a smaller number of epochs for comparison
    TRAINING_CONFIG['epochs'] = epochs
    print(f"Training each model for {epochs} epochs")
    
    # Load data
    processor = ChessDataProcessor()
    
    if data_dir:
        DATA_CONFIG['processed_dir'] = data_dir
    
    try:
        train_data = processor.load_processed_data("train_data.pkl")
        test_data = processor.load_processed_data("test_data.pkl")
    except FileNotFoundError:
        print("No preprocessed data found. Running data preprocessing pipeline...")
        train_data, test_data = processor.run_pipeline()
    
    print(f"Loaded data: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Create datasets and dataloaders
    train_dataset = ChessConceptDataset(train_data)
    test_dataset = ChessConceptDataset(test_data)
    
    # Train and evaluate each model
    model_results = {}
    training_times = {}
    evaluation_results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        # Create model directory
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        TRAINING_CONFIG['checkpoint_dir'] = model_dir
        
        # Create trainer
        trainer = ConceptTrainer(TRAINING_CONFIG)
        
        # Train model
        start_time = time.time()
        model, history = trainer.train(train_data, test_data, model_type)
        training_time = time.time() - start_time
        
        # Save training time
        training_times[model_type] = training_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Create test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=False
        )
        
        # Create evaluator
        evaluator = ConceptEvaluator()
        
        # Evaluate model
        print(f"Evaluating {model_type} model...")
        metrics = evaluator.evaluate(model, test_loader)
        
        # Save metrics
        with open(os.path.join(model_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Store results
        evaluation_results[model_type] = metrics
        model_results[model_type] = {
            'training_time': training_time,
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
        }
        
        # Plot training history
        history_path = os.path.join(model_dir, 'training_history.json')
        plot_training_history(history_path)
    
    # Compare models
    print("\nModel Comparison Results:")
    print("-----------------------")
    
    comparison_data = []
    headers = ["Model Type", "Training Time (s)", "Best Val Loss", "F1 Score", "Accuracy"]
    
    for model_type in model_types:
        metrics = evaluation_results[model_type]
        results = model_results[model_type]
        
        f1_score = metrics['macro_avg']['f1']
        accuracy = metrics['overall_accuracy']
        
        comparison_data.append([
            model_type,
            f"{results['training_time']:.2f}",
            f"{results['best_val_loss']:.4f}",
            f"{f1_score:.4f}",
            f"{accuracy:.4f}"
        ])
        
        print(f"{model_type}: Training Time={results['training_time']:.2f}s, "
              f"Val Loss={results['best_val_loss']:.4f}, F1={f1_score:.4f}, "
              f"Accuracy={accuracy:.4f}")
    
    # Save comparison table
    with open(os.path.join(output_dir, 'model_comparison.csv'), 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in comparison_data:
            f.write(','.join(row) + '\n')
    
    # Visualize comparison
    visualize_model_comparison(
        {m_type: evaluation_results[m_type] for m_type in model_types},
        metric='f1',
        save_path=os.path.join(output_dir, 'model_comparison_f1.png'),
        show=False
    )
    
    visualize_model_comparison(
        {m_type: evaluation_results[m_type] for m_type in model_types},
        metric='precision',
        save_path=os.path.join(output_dir, 'model_comparison_precision.png'),
        show=False
    )
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(model_types, [model_results[m]['training_time'] for m in model_types])
    plt.title('Training Time Comparison')
    plt.xlabel('Model Architecture')
    plt.ylabel('Training Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    
    print(f"\nComparison results saved to {output_dir}")
    return evaluation_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and compare different model architectures.")
    parser.add_argument("--data-dir", type=str, help="Directory with processed data")
    parser.add_argument("--output-dir", type=str, default="model_comparison_results",
                        help="Directory to save results")
    parser.add_argument("--models", type=str, nargs='+', 
                        choices=['cnn', 'transformer', 'nnue_based'],
                        default=['cnn', 'transformer', 'nnue_based'],
                        help="Models to train and compare")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train each model")
    
    args = parser.parse_args()
    train_and_compare_models(args.data_dir, args.output_dir, args.models, args.epochs)