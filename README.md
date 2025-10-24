# Transformer Model Implementation from Scratch

A complete implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) using TensorFlow/Keras.

## Overview

This implementation provides a modular and educational version of the Transformer model, which revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. The code is structured to be easily understandable while maintaining the full functionality of the original architecture.

## Features

- **Multi-Head Self-Attention**: Parallel attention mechanisms to capture different linguistic relationships
- **Positional Encoding**: Sinusoidal positional embeddings to provide sequence order information
- **Encoder-Decoder Architecture**: Standard transformer architecture with multiple layers
- **Masking Support**: Look-ahead masking for decoder and padding masking for variable-length sequences
- **Modular Design**: Each component is implemented as a separate, reusable layer
- **Educational Focus**: Clean, well-commented code suitable for learning and experimentation

## Architecture Components

### 1. Positional Encoding
- Implements sinusoidal positional embeddings
- Handles sequences of variable length up to `max_len`
- Combines sine and cosine functions of different frequencies

### 2. Multi-Head Attention
- Scaled dot-product attention mechanism
- Parallel attention heads for different representation subspaces
- Support for different types of masking

### 3. Feed-Forward Network
- Position-wise fully connected feed-forward network
- ReLU activation in hidden layer
- Residual connections and layer normalization

### 4. Encoder
- Stack of identical encoder layers
- Self-attention and feed-forward components
- Residual connections and dropout for regularization

### 5. Decoder
- Stack of identical decoder layers
- Masked self-attention and encoder-decoder attention
- Three sub-layers per decoder layer

## Installation & Dependencies

```bash
pip install tensorflow numpy
