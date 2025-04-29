"""
Evaluation module for BigBadCollie.
Handles model evaluation, metrics calculation, and visualization.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import json
import chess
import chess.svg

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import EVAL_CONFIG, ALL_CONCEPTS


class ConceptEvaluator:
    """
    Evaluator for BigBadCollie chess concept recognition model.
    """
    
    def __init__(self, config=EVAL_CONFIG):
        """Initialize evaluator with configuration."""
        self.config = config
        self.threshold = config['threshold']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs('evaluation_results', exist_ok=True)
    
    def evaluate(self, model, test_loader):
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_outputs = []
        all_labels = []
        
        # Get predictions and labels
        with torch.no_grad():
            for batch in test_loader:
                boards = batch['board'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(boards).cpu().numpy()
                
                all_outputs.extend(outputs)
                all_labels.extend(labels)
        
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_outputs, all_labels)
        
        # Save results
        self.save_results(metrics)
        
        return metrics
    
    def calculate_metrics(self, outputs, labels):
        """
        Calculate evaluation metrics.
        
        Args:
            outputs: Model outputs (probabilities)
            labels: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions using threshold
        predictions = (outputs >= self.threshold).astype(int)
        
        # Calculate precision, recall, and F1 score for each concept
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Calculate macro-averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # Calculate micro-averaged metrics
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro'
        )
        
        # Calculate weighted-averaged metrics
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Calculate accuracy for each concept
        accuracy = np.mean((predictions == labels).astype(int), axis=0)
        
        # Calculate overall accuracy
        overall_accuracy = np.mean((predictions == labels).all(axis=1))
        
        # Calculate AUC-ROC for each concept
        auc_scores = []
        for i in range(labels.shape[1]):
            try:
                fpr, tpr, _ = roc_curve(labels[:, i], outputs[:, i])
                auc_scores.append(auc(fpr, tpr))
            except:
                auc_scores.append(0)
        
        # Create metrics dictionary
        metrics = {
            'per_concept': {
                concept: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'accuracy': float(accuracy[i]),
                    'auc': float(auc_scores[i])
                }
                for i, concept in enumerate(ALL_CONCEPTS)
            },
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1)
            },
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1': float(micro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1': float(weighted_f1)
            },
            'overall_accuracy': float(overall_accuracy)
        }
        
        return metrics
    
    def save_results(self, metrics, filename='evaluation_metrics.json'):
        """Save evaluation metrics to file."""
        save_path = os.path.join('evaluation_results', filename)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to {save_path}")
    
    def plot_confusion_matrix(self, outputs, labels, save_path='evaluation_results/confusion_matrix.png'):
        """Plot confusion matrix for each concept."""
        if not self.config.get('confusion_matrix', False):
            return
        
        predictions = (outputs >= self.threshold).astype(int)
        num_concepts = len(ALL_CONCEPTS)
        
        plt.figure(figsize=(20, 20))
        for i, concept in enumerate(ALL_CONCEPTS):
            plt.subplot(int(np.ceil(num_concepts / 4)), 4, i + 1)
            cm = confusion_matrix(labels[:, i], predictions[:, i])
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix: {concept}')
            plt.colorbar()
            tick_marks = [0, 1]
            plt.xticks(tick_marks, ['Negative', 'Positive'])
            plt.yticks(tick_marks, ['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Add text annotations
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    def visualize_board_concepts(self, model, data_loader, num_samples=None):
        """
        Visualize chess boards with predicted concepts.
        
        Args:
            model: Trained model
            data_loader: DataLoader for test data
            num_samples: Number of samples to visualize (default: from config)
        """
        if not num_samples:
            num_samples = self.config.get('visualization_samples', 5)
        
        model.eval()
        
        # Create output directory
        vis_dir = os.path.join('evaluation_results', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get samples
        samples = []
        with torch.no_grad():
            for batch in data_loader:
                boards = batch['board'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(boards).cpu().numpy()
                
                for i in range(len(boards)):
                    board_array = boards[i].cpu().numpy()
                    true_labels = labels[i]
                    pred_probs = outputs[i]
                    
                    samples.append((board_array, true_labels, pred_probs))
                    
                    if len(samples) >= num_samples:
                        break
                
                if len(samples) >= num_samples:
                    break
        
        # Visualize samples
        for i, (board_array, true_labels, pred_probs) in enumerate(samples):
            # Convert board array to chess.Board
            board = self._array_to_board(board_array)
            
            # Convert board to SVG image
            svg = chess.svg.board(board, size=350)
            png_path = os.path.join(vis_dir, f'board_{i}.png')
            
            # Convert SVG to PNG
            self._svg_to_png(svg, png_path)
            
            # Create concept prediction visualization
            pred_labels = (pred_probs >= self.threshold).astype(int)
            self._visualize_concept_predictions(
                true_labels, pred_labels, pred_probs,
                os.path.join(vis_dir, f'concepts_{i}.png')
            )
            
            print(f"Visualized sample {i+1}/{num_samples}")
    
    def _array_to_board(self, board_array):
        """
        Convert board array representation to chess.Board object.
        
        Args:
            board_array: 8x8x12 numpy array
            
        Returns:
            chess.Board object
        """
        board = chess.Board.empty()
        
        piece_types = [
            # White pieces (indices 0-5)
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING,
            # Black pieces (indices 6-11)
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        colors = [chess.WHITE] * 6 + [chess.BLACK] * 6
        
        for row in range(8):
            for col in range(8):
                for piece_idx in range(12):
                    if board_array[row, col, piece_idx] > 0:
                        square = chess.square(col, 7 - row)  # Convert to chess.Square
                        piece = chess.Piece(piece_types[piece_idx], colors[piece_idx])
                        board.set_piece_at(square, piece)
        
        return board
    
    def _svg_to_png(self, svg_data, output_path, dpi=100):
        """
        Convert SVG to PNG image.
        
        Note: This function requires cairosvg library.
        If not available, it will skip the conversion.
        """
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=output_path, dpi=dpi)
        except ImportError:
            print("Warning: cairosvg not available, skipping SVG to PNG conversion")
            # Write the SVG to file instead
            with open(output_path.replace('.png', '.svg'), 'w') as f:
                f.write(svg_data)
    
    def _visualize_concept_predictions(self, true_labels, pred_labels, pred_probs, output_path):
        """
        Visualize concept predictions.
        
        Args:
            true_labels: True concept labels
            pred_labels: Predicted concept labels
            pred_probs: Prediction probabilities
            output_path: Output file path
        """
        # Sort concepts by prediction probability
        sorted_indices = np.argsort(pred_probs)[::-1]
        
        plt.figure(figsize=(12, 8))
        
        # Bar chart of concept probabilities
        plt.subplot(1, 1, 1)
        x = np.arange(len(ALL_CONCEPTS))
        bar_width = 0.35
        
        # Plot bars
        plt.bar(x - bar_width/2, true_labels, bar_width, label='True', alpha=0.7)
        plt.bar(x + bar_width/2, pred_probs, bar_width, label='Predicted', alpha=0.7)
        
        # Add labels and ticks
        plt.xlabel('Chess Concepts')
        plt.ylabel('Probability')
        plt.title('Concept Predictions')
        plt.xticks(x, [ALL_CONCEPTS[i] for i in sorted_indices], rotation=90)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    # This would be run from the main evaluation script
    pass