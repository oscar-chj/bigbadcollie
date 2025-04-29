#!/usr/bin/env python3
"""
Example script to demonstrate BigBadCollie's concept recognition on a chess position.
"""
import os
import sys
import torch
import chess
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BigBadCollie modules
from config.config import ALL_CONCEPTS, MODEL_CONFIG, EVAL_CONFIG
from data.processor import ChessDataProcessor
from models.model import create_model
from utils.visualize import render_chess_board


def analyze_position(fen=None, model_path=None, threshold=0.5):
    """
    Analyze a chess position with BigBadCollie model.
    
    Args:
        fen: FEN string representation of a position (uses starting position if None)
        model_path: Path to a trained model checkpoint (uses a dummy model if None)
        threshold: Threshold for concept detection
    """
    # Create a chess board from FEN
    if fen is None:
        # Use a slightly more interesting position than the starting position
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    board = chess.Board(fen)
    print(f"Analyzing position: {fen}")
    
    # Create a board representation
    processor = ChessDataProcessor()
    board_array = processor.board_to_array(board)
    
    # Visualize the board
    render_chess_board(board, show=False, save_path="example_position.png")
    print("Board visualization saved as 'example_position.png'")
    
    # Create or load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path and os.path.exists(model_path):
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        model_type = checkpoint['config']['base_model']
        model = create_model(model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        # Create a dummy model (for demonstration)
        print("No model checkpoint provided, using an untrained model (results will be random)")
        model = create_model(MODEL_CONFIG['base_model'])
    
    model.to(device)
    model.eval()
    
    # Prepare input tensor
    board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict concepts
    with torch.no_grad():
        predictions = model(board_tensor).cpu().numpy()[0]
    
    # Display predictions with confidence above threshold
    print("\nDetected Chess Concepts:")
    print("-----------------------")
    
    # Group concepts by category
    concept_results = {}
    for i, concept in enumerate(ALL_CONCEPTS):
        if predictions[i] >= threshold:
            # Determine which category this concept belongs to
            category = None
            for cat, concepts in processor.config.get('CHESS_CONCEPTS', {}).items():
                if concept in concepts:
                    category = cat
                    break
            
            if category not in concept_results:
                concept_results[category] = []
            
            concept_results[category].append((concept, predictions[i]))
    
    # Print results by category
    if not concept_results:
        print("No concepts detected above threshold of {threshold}")
    else:
        for category in ["positional", "tactical", "strategic"]:
            if category in concept_results:
                print(f"\n{category.capitalize()} concepts:")
                for concept, confidence in sorted(concept_results[category], key=lambda x: x[1], reverse=True):
                    print(f"  - {concept.replace('_', ' ').title()}: {confidence:.2f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    
    # Sort concepts by prediction value
    sorted_indices = predictions.argsort()[::-1]
    sorted_concepts = [ALL_CONCEPTS[i] for i in sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    # Plot bar chart
    bars = plt.bar(range(len(sorted_concepts)), sorted_predictions)
    
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if sorted_predictions[i] >= threshold:
            bar.set_color('green')
        else:
            bar.set_color('grey')
            
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xticks(range(len(sorted_concepts)), [c.replace('_', '\n') for c in sorted_concepts], rotation=90)
    plt.ylabel('Confidence')
    plt.title('Chess Concepts Detected in Position')
    plt.tight_layout()
    plt.savefig("example_concepts.png")
    print("Concept analysis visualization saved as 'example_concepts.png'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a chess position for concepts.")
    parser.add_argument("--fen", type=str, help="FEN string of the position to analyze")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    analyze_position(args.fen, args.model, args.threshold)