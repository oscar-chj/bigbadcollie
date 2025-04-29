# BigBadCollie 🧠♟️

## Neural Network for Chess Concept Recognition

BigBadCollie (BBC) is a neural network model that analyzes chess moves to identify the underlying human concepts—such as "center control," "king safety," or "pawn structure"—rather than merely determining the optimal move.

## Project Objectives

- Identify human-understandable chess concepts in positions and moves
- Bridge the gap between engine evaluation and human understanding
- Create a tool that can help annotate games with conceptual insights

## Features

- Multiple neural network architectures (CNN, Transformer, NNUE-based)
- Recognition of 24 different chess concepts across tactical, positional, and strategic categories
- Support for analyzing individual positions (FEN) or entire games (PGN)
- Visualization tools for model interpretation

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install cairosvg for visualization support:

```bash
pip install cairosvg
```

## Usage

### Data Preprocessing

To preprocess PGN files for training:

```bash
python main.py preprocess --pgn-dir path/to/pgn/files
```

### Model Training

To train a model:

```bash
python main.py train --model-type cnn
```

### Model Evaluation

To evaluate a trained model:

```bash
python main.py evaluate --visualize
```

### Concept Prediction

To predict concepts for a specific position:

```bash
python main.py predict --fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
```

Or for an entire game:

```bash
python main.py predict --pgn-file path/to/game.pgn
```

## Project Structure

- `config/` - Configuration files and parameters
- `data/` - Data processing utilities
- `models/` - Neural network model architectures
- `training/` - Training pipeline
- `evaluation/` - Evaluation and visualization tools
- `utils/` - Utility functions

## Chess Concepts

The model is trained to recognize the following concepts:

### Positional
- Center control
- Pawn structure
- Piece activity
- King safety
- Space advantage
- Outpost
- Weak squares
- Piece coordination

### Tactical
- Fork
- Pin
- Skewer
- Discovered attack
- Double attack
- Zugzwang
- Overloaded piece
- Sacrifice

### Strategic
- Pawn majority
- Isolated pawn
- Passed pawn
- Backward pawn
- Queenside attack
- Kingside attack
- Minority attack
- Prophylaxis

## Acknowledgements

This project builds upon research in:
- Maia Chess Engine
- AlphaZero's concept learning capabilities
- CHREST cognitive architecture