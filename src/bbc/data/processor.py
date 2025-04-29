"""
Data processing module for BigBadCollie.
Handles loading, processing and preparing chess game data for training.
"""
import os
import chess
import chess.pgn
import numpy as np
from typing import List, Dict, Tuple
import pickle
import sys

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_CONFIG, CHESS_CONCEPTS, ALL_CONCEPTS

class ChessDataProcessor:
    """
    Processes chess PGN files and prepares data for training the concept recognition model.
    """
    
    def __init__(self, config=DATA_CONFIG):
        """Initialize the data processor with configuration."""
        self.config = config
        self.pgn_dir = config['pgn_dir']
        self.processed_dir = config['processed_dir']
        self.games_limit = config['games_limit']
        self.min_elo = config['min_elo']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def board_to_array(self, board: chess.Board) -> np.ndarray:
        """
        Convert a chess board to a numpy array representation.
        
        Args:
            board: A chess.Board object
            
        Returns:
            np.ndarray: 8x8x12 array representation of the board
        """
        # Create an empty 8x8x12 array (12 = 6 piece types * 2 colors)
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Piece types (pawn, knight, bishop, rook, queen, king) for each color
        pieces = [
            # White pieces (indices 0-5)
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING,
            # Black pieces (indices 6-11)
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        colors = [chess.WHITE] * 6 + [chess.BLACK] * 6
        
        # Fill the board representation
        for i in range(64):
            row, col = i // 8, i % 8
            piece = board.piece_at(i)
            if piece:
                # Find the index for this piece
                for j, (p_type, color) in enumerate(zip(pieces, colors)):
                    if piece.piece_type == p_type and piece.color == color:
                        board_array[row, col, j] = 1
                        break
                        
        return board_array
    
    def extract_features_from_game(self, game: chess.pgn.Game) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract board positions and concepts from a chess game.
        
        Args:
            game: A chess.pgn.Game object
            
        Returns:
            List of tuples (board_array, concepts_dict)
        """
        # Skip games if either player has low rating
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        
        if white_elo < self.min_elo or black_elo < self.min_elo:
            return []
        
        # Extract board positions and concepts
        data = []
        board = game.board()
        
        # Get annotated concepts if available
        comments_map = self._extract_comments(game)
        
        for move_idx, move in enumerate(game.mainline_moves()):
            board.push(move)
            board_array = self.board_to_array(board)
            
            # Determine concepts for this position/move
            # In real implementation, these would come from annotations or engine analysis
            concepts = self._extract_concepts_from_comments(comments_map.get(move_idx, ""))
            
            data.append((board_array, concepts))
            
        return data
    
    def _extract_comments(self, game: chess.pgn.Game) -> Dict[int, str]:
        """Extract comments from a game indexed by move number."""
        comments = {}
        node = game
        move_idx = -1
        
        while node.variations:
            next_node = node.variation(0)
            move_idx += 1
            if next_node.comment:
                comments[move_idx] = next_node.comment
            node = next_node
            
        return comments
    
    def _extract_concepts_from_comments(self, comment: str) -> Dict[str, float]:
        """
        Extract chess concepts from move comments.
        
        This is a placeholder - in a real implementation, you would use NLP
        or pattern matching to identify concepts in the comments.
        
        Args:
            comment: Comment text for a move
            
        Returns:
            Dict mapping concept names to confidence scores (0-1)
        """
        concepts = {concept: 0.0 for concept in ALL_CONCEPTS}
        
        # Simple keyword matching for demo purposes
        comment = comment.lower()
        
        for concept in ALL_CONCEPTS:
            # Replace underscores with spaces for matching
            concept_term = concept.replace('_', ' ')
            if concept_term in comment:
                # Set confidence based on context
                concepts[concept] = 0.9
                
                # Look for modifiers
                if "strong" in comment or "clear" in comment:
                    concepts[concept] = 0.95
                elif "slight" in comment or "minor" in comment:
                    concepts[concept] = 0.7
                    
        return concepts
    
    def process_pgn_file(self, pgn_file: str) -> List[Tuple[np.ndarray, Dict]]:
        """Process a single PGN file and extract training data."""
        data = []
        games_processed = 0
        
        try:
            with open(pgn_file, 'r') as f:
                while games_processed < self.games_limit:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    game_data = self.extract_features_from_game(game)
                    data.extend(game_data)
                    games_processed += 1
                    
                    if games_processed % 100 == 0:
                        print(f"Processed {games_processed} games")
        except Exception as e:
            print(f"Error processing {pgn_file}: {e}")
            
        return data
    
    def process_all_pgn_files(self):
        """Process all PGN files in the data directory."""
        all_data = []
        
        # Get all PGN files
        pgn_files = [os.path.join(self.pgn_dir, f) for f in os.listdir(self.pgn_dir) 
                    if f.endswith('.pgn')]
        
        for pgn_file in pgn_files:
            print(f"Processing {pgn_file}...")
            data = self.process_pgn_file(pgn_file)
            all_data.extend(data)
            
            if len(all_data) >= self.games_limit * 40:  # Assuming ~40 positions per game
                break
        
        return all_data
    
    def save_processed_data(self, data, filename: str = "chess_data.pkl"):
        """Save processed data to disk."""
        save_path = os.path.join(self.processed_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {save_path}")
        
    def load_processed_data(self, filename: str = "chess_data.pkl"):
        """Load processed data from disk."""
        load_path = os.path.join(self.processed_dir, filename)
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def prepare_train_test_split(self, data):
        """Split data into training and testing sets."""
        # Shuffle data
        np.random.shuffle(data)
        
        # Split into train/test
        split_idx = int(len(data) * (1 - self.config['test_split']))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        return train_data, test_data

    def run_pipeline(self):
        """Run the full data processing pipeline."""
        print("Starting data processing pipeline...")
        
        # Check if processed data already exists
        processed_file = os.path.join(self.processed_dir, "chess_data.pkl")
        if os.path.exists(processed_file):
            print(f"Found existing processed data at {processed_file}")
            data = self.load_processed_data()
        else:
            # Process PGN files
            data = self.process_all_pgn_files()
            self.save_processed_data(data)
        
        # Prepare train/test split
        train_data, test_data = self.prepare_train_test_split(data)
        
        # Save splits
        self.save_processed_data(train_data, "train_data.pkl")
        self.save_processed_data(test_data, "test_data.pkl")
        
        print(f"Data processing complete. Train size: {len(train_data)}, Test size: {len(test_data)}")
        return train_data, test_data


if __name__ == "__main__":
    processor = ChessDataProcessor()
    processor.run_pipeline()