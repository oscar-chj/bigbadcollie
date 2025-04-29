"""
Configuration file for BigBadCollie project.
Defines chess concepts, model parameters, and training settings.
"""

# Chess concepts to recognize
CHESS_CONCEPTS = {
    'positional': [
        'center_control',
        'pawn_structure',
        'piece_activity',
        'king_safety',
        'space_advantage',
        'outpost',
        'weak_squares',
        'piece_coordination'
    ],
    'tactical': [
        'fork',
        'pin',
        'skewer',
        'discovered_attack',
        'double_attack',
        'zugzwang',
        'overloaded_piece',
        'sacrifice'
    ],
    'strategic': [
        'pawn_majority',
        'isolated_pawn',
        'passed_pawn',
        'backward_pawn',
        'queenside_attack',
        'kingside_attack',
        'minority_attack',
        'prophylaxis'
    ]
}

# Flatten concepts for model output
ALL_CONCEPTS = []
for category in CHESS_CONCEPTS:
    ALL_CONCEPTS.extend(CHESS_CONCEPTS[category])

# Model architecture parameters
MODEL_CONFIG = {
    'base_model': 'cnn',  # Options: 'cnn', 'transformer', 'nnue_based'
    'input_shape': (8, 8, 12),  # 8x8 board with 12 piece types (6 pieces × 2 colors)
    'conv_filters': [64, 128, 256],
    'conv_kernel_size': 3,
    'fc_layers': [512, 256],
    'dropout_rate': 0.3,
    'num_concepts': len(ALL_CONCEPTS),
    'activation': 'relu',
    'output_activation': 'sigmoid',  # For multi-label classification
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'checkpoint_dir': 'checkpoints',
    'use_augmentation': True,
    'optimizer': 'adam',
}

# Data parameters
DATA_CONFIG = {
    'pgn_dir': 'data/pgn',
    'processed_dir': 'data/processed',
    'games_limit': 100000,  # Max number of games to process
    'min_elo': 2000,  # Minimum player Elo rating for training data
    'annotation_source': 'stockfish',  # Options: 'stockfish', 'human', 'combined'
    'stockfish_depth': 18,
    'test_split': 0.1,
}

# Evaluation parameters
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'threshold': 0.5,  # Probability threshold for concept detection
    'visualization_samples': 20,
    'confusion_matrix': True,
}

# NNUE integration parameters
NNUE_CONFIG = {
    'nnue_model_path': None,  # Path to pretrained NNUE model if using NNUE-based architecture
    'feature_transformer_layers': 2,
    'use_halfkp': True,  # Use HalfKP features from NNUE
    'train_from_scratch': False,  # Whether to train NNUE features from scratch
}