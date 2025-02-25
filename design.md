# Character-Level Transformer Model and Correlation Analysis

This project implements a character-level transformer model for sequence prediction and analyzes character correlations within predicted sequences.

## Project Structure

The project consists of two main components:
1. Training code (`training.py`)
2. Correlation analysis code (`correlation.py`)

### Training Code Overview

The training code implements a character-level transformer model with the following key components:

- **Data Generation**: Creates random character sequences using ASCII letters, digits, and punctuation
- **CharacterDataset**: Custom PyTorch dataset for character sequence handling
- **CharacterTransformer**: Main model architecture featuring:
  - Embedding layer
  - Positional encoding
  - Transformer encoder layers
  - Final linear layer for prediction

Training parameters:
- Sequence length: 50 characters
- Batch size: 64
- Learning rate: 0.001
- Number of epochs: 10
- Model dimensions: 128
- Number of attention heads: 8
- Number of transformer layers: 3

### Correlation Analysis

The correlation analysis code examines how characters in a sequence influence the prediction of subsequent characters. Key features:

- Analyzes how each character affects the prediction of the following character
- Measures correlation strength by varying individual characters and observing changes in prediction probabilities
- Provides ranked correlation strengths between characters

## Example Usage

```python
# Training
sequence = generate_char_sequence(400000, 128)
model = CharacterTransformer(vocab_size)
trained_model = train_model(model, sequence, char_to_idx)

# Correlation Analysis
sample_text = "Hello123"
get_last_char_correlation(model, sample_text, char_to_idx)
```

## Purpose of Experiment

The experiment aims to:
1. Train a transformer model to understand character-level patterns in sequences
2. Analyze how different characters influence the model's predictions
3. Quantify the correlation between characters in sequence prediction
4. Understand the context window usage in transformer models

This experiment helps visualize how transformer models utilize context for predictions and how different positions in the input sequence affect the output probabilities.

## Dependencies

- PyTorch
- Matplotlib
- NumPy
- String (Python standard library)
=======
# attention

