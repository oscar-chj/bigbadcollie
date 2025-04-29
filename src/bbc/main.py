#!/usr/bin/env python3
"""
BigBadCollie: A neural network model for identifying human chess concepts.

This is the main entry point for the BigBadCollie project, which aims to recognize
chess concepts (like "center control", "king safety", or "pawn structure") from
chess positions and moves.
"""

import os
import sys
import argparse
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bbc.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("BigBadCollie")

# Import project modules
from config.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, ALL_CONCEPTS
from data.processor import ChessDataProcessor
from models.model import create_model
from training.trainer import ConceptTrainer, ChessConceptDataset
from evaluation.evaluator import ConceptEvaluator
from torch.utils.data import DataLoader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BigBadCollie - Chess Concept Recognition")
    
    # Main action to perform
    parser.add_argument('action', type=str, choices=['preprocess', 'train', 'evaluate', 'predict'],
                        help='Action to perform')
    
    # Data-related arguments
    parser.add_argument('--pgn-dir', type=str, help='Directory containing PGN files')
    parser.add_argument('--data-dir', type=str, help='Directory for processed data')
    parser.add_argument('--games-limit', type=int, help='Maximum number of games to process')
    
    # Model-related arguments
    parser.add_argument('--model-type', type=str, choices=['cnn', 'transformer', 'nnue_based'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    
    # Training-related arguments
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Evaluation-related arguments
    parser.add_argument('--threshold', type=float, help='Prediction threshold')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    # Prediction-related arguments
    parser.add_argument('--fen', type=str, help='FEN string for prediction')
    parser.add_argument('--pgn-file', type=str, help='PGN file for prediction')
    
    # Miscellaneous
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration dictionaries from command-line arguments."""
    # Update DATA_CONFIG
    if args.pgn_dir:
        DATA_CONFIG['pgn_dir'] = args.pgn_dir
    if args.data_dir:
        DATA_CONFIG['processed_dir'] = args.data_dir
    if args.games_limit:
        DATA_CONFIG['games_limit'] = args.games_limit
    
    # Update MODEL_CONFIG
    if args.model_type:
        MODEL_CONFIG['base_model'] = args.model_type
    
    # Update TRAINING_CONFIG
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.learning_rate:
        TRAINING_CONFIG['learning_rate'] = args.learning_rate
    
    # Update EVAL_CONFIG
    if args.threshold:
        EVAL_CONFIG['threshold'] = args.threshold


def preprocess_data(args):
    """Preprocess chess data."""
    logger.info("Starting data preprocessing...")
    
    # Create data processor
    processor = ChessDataProcessor(DATA_CONFIG)
    
    # Run preprocessing pipeline
    train_data, test_data = processor.run_pipeline()
    
    logger.info(f"Preprocessing complete. Train size: {len(train_data)}, Test size: {len(test_data)}")
    return train_data, test_data


def train_model(args, train_data=None, val_data=None):
    """Train the chess concept recognition model."""
    logger.info("Starting model training...")
    
    # If data is not provided, load from disk
    if train_data is None or val_data is None:
        processor = ChessDataProcessor(DATA_CONFIG)
        try:
            train_data = processor.load_processed_data("train_data.pkl")
            val_data = processor.load_processed_data("test_data.pkl")
        except FileNotFoundError:
            logger.error("No preprocessed data found. Run with 'preprocess' action first.")
            sys.exit(1)
    
    # Create trainer
    trainer = ConceptTrainer(TRAINING_CONFIG)
    
    # Train model
    model_type = args.model_type if args.model_type else MODEL_CONFIG['base_model']
    model, history = trainer.train(train_data, val_data, model_type)
    
    logger.info(f"Training completed. Best validation loss: {history['best_val_loss']:.4f}")
    return model, history


def evaluate_model(args):
    """Evaluate a trained model."""
    logger.info("Starting model evaluation...")
    
    # Load test data
    processor = ChessDataProcessor(DATA_CONFIG)
    try:
        test_data = processor.load_processed_data("test_data.pkl")
    except FileNotFoundError:
        logger.error("No test data found. Run with 'preprocess' action first.")
        sys.exit(1)
    
    # Create test dataset and loader
    test_dataset = ChessConceptDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False
    )
    
    # Load model
    if not args.checkpoint:
        checkpoint_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], "best_model.pt")
        if not os.path.exists(checkpoint_path):
            logger.error("No model checkpoint found. Train a model first.")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model_type = checkpoint['config']['base_model']
    model = create_model(model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create evaluator
    evaluator = ConceptEvaluator(EVAL_CONFIG)
    
    # Evaluate model
    metrics = evaluator.evaluate(model, test_loader)
    
    # Print summary metrics
    logger.info("Evaluation complete. Summary metrics:")
    logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"Macro F1 score: {metrics['macro_avg']['f1']:.4f}")
    logger.info(f"Micro F1 score: {metrics['micro_avg']['f1']:.4f}")
    
    # Visualize results if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        evaluator.visualize_board_concepts(model, test_loader)
    
    return metrics


def predict_concepts(args):
    """
    Predict chess concepts for a position or game.
    """
    logger.info("Starting concept prediction...")
    
    # Load model
    if not args.checkpoint:
        checkpoint_path = os.path.join(TRAINING_CONFIG['checkpoint_dir'], "best_model.pt")
        if not os.path.exists(checkpoint_path):
            logger.error("No model checkpoint found. Train a model first.")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model_type = checkpoint['config']['base_model']
    model = create_model(model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process input
    processor = ChessDataProcessor(DATA_CONFIG)
    
    if args.fen:
        # Convert FEN to board representation
        import chess
        board = chess.Board(args.fen)
        board_array = processor.board_to_array(board)
        board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict concepts
        with torch.no_grad():
            predictions = model(board_tensor).cpu().numpy()[0]
        
        # Display predictions
        print("\nConcept predictions for position:")
        print("--------------------------------")
        for i, concept in enumerate(ALL_CONCEPTS):
            if predictions[i] >= EVAL_CONFIG['threshold']:
                print(f"{concept}: {predictions[i]:.4f}")
        
    elif args.pgn_file:
        # Process PGN file
        import chess.pgn
        
        try:
            with open(args.pgn_file, 'r') as f:
                game = chess.pgn.read_game(f)
                if game is None:
                    logger.error("Could not read game from PGN file.")
                    return
                
                # Extract features from game
                game_data = processor.extract_features_from_game(game)
                
                if not game_data:
                    logger.error("No valid positions found in game.")
                    return
                
                # Predict concepts for each position
                results = []
                for i, (board_array, _) in enumerate(game_data):
                    board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        predictions = model(board_tensor).cpu().numpy()[0]
                    
                    # Get top concepts
                    top_concepts = [(ALL_CONCEPTS[j], predictions[j]) 
                                   for j in range(len(ALL_CONCEPTS))
                                   if predictions[j] >= EVAL_CONFIG['threshold']]
                    top_concepts.sort(key=lambda x: x[1], reverse=True)
                    
                    results.append((i+1, top_concepts))
                
                # Display results
                print("\nConcept analysis for game:")
                print("-------------------------")
                for move_num, concepts in results:
                    if concepts:
                        print(f"Move {move_num}: {', '.join([f'{c} ({p:.2f})' for c, p in concepts[:3]])}")
                
        except Exception as e:
            logger.error(f"Error processing PGN file: {e}")
    else:
        logger.error("Either --fen or --pgn-file must be provided for prediction.")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Update configs from command-line arguments
    update_config_from_args(args)
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Execute the requested action
    if args.action == 'preprocess':
        preprocess_data(args)
    elif args.action == 'train':
        train_model(args)
    elif args.action == 'evaluate':
        evaluate_model(args)
    elif args.action == 'predict':
        predict_concepts(args)
    

if __name__ == "__main__":
    main()