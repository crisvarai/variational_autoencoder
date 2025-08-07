# Variational AutoEncoder (VAE) with PyTorch

A PyTorch implementation of a Variational AutoEncoder for generating handwritten digits using the MNIST dataset. This project includes both a modular Python implementation and a comprehensive Jupyter notebook for experimentation.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Configuration](#configuration)

## Features

- **Variational AutoEncoder Implementation**: Complete VAE architecture with encoder-decoder structure
- **MNIST Dataset Support**: Automatic download and preprocessing of MNIST handwritten digits
- **Flexible Training Pipeline**: Configurable hyperparameters and training settings
- **Image Generation**: Generate new handwritten digits from learned latent representations
- **Modular Design**: Clean, reusable code structure with separate modules
- **Logging Support**: Comprehensive logging for training progress and model checkpoints
- **GPU Support**: Automatic CUDA detection and usage when available

## Project Structure

```
├── main.py                          # Main training script
├── requirements.txt                 # Python dependencies
├── VAE_pytorch_50_epochs.ipynb     # Jupyter notebook for experimentation
├── model/
│   └── VAE.py                      # VAE model implementation
├── train/
│   ├── fit.py                      # Training loop implementation
│   └── inference.py                # Inference and generation functions
├── utils/
│   ├── load_args.py               # Command line argument parsing
│   └── data.py                    # Data loading and preprocessing utilities
├── weights/                       # Directory for saved model checkpoints
└── generated/                     # Directory for generated images
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vae-pytorch
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv vae_env
   source vae_env/bin/activate  # On Windows: vae_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Training

Run the training script with default parameters:

```bash
python main.py
```

### Custom Training

Train with custom hyperparameters:

```bash
python main.py --epochs 100 --batchsize 64 --lr 1e-3 --zdim 32
```

### Using Jupyter Notebook

Launch the interactive notebook for experimentation:

```bash
jupyter notebook VAE_pytorch_50_epochs.ipynb
```

## Usage

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | "./" | Path to store MNIST dataset |
| `--indim` | int | 784 | Input dimension (28x28 flattened) |
| `--hdim` | int | 200 | Hidden layer dimension |
| `--zdim` | int | 20 | Latent space dimension |
| `--epochs` | int | 51 | Number of training epochs |
| `--batchsize` | int | 32 | Training batch size |
| `--lr` | float | 3e-4 | Learning rate |
| `--wgts_path` | str | "weights/best_model_py_project.pth" | Model checkpoint save path |

### Example Usage

```bash
# Train a deeper latent space model
python main.py --zdim 50 --hdim 400 --epochs 100

# Train with larger batch size and higher learning rate
python main.py --batchsize 128 --lr 1e-3

# Specify custom data and weights paths
python main.py --data_path ./data --wgts_path ./checkpoints/my_model.pth
```

## Model Architecture

The VAE consists of two main components:

### Encoder
- **Input**: Flattened 28×28 MNIST images (784 dimensions)
- **Hidden Layer**: Fully connected layer with ReLU activation
- **Output**: Two branches for mean (μ) and log variance (σ) of latent distribution

### Decoder  
- **Input**: Sampled latent vector from N(μ, σ²)
- **Hidden Layer**: Fully connected layer with ReLU activation
- **Output**: Reconstructed image with sigmoid activation

### Loss Function
The VAE loss combines two components:
- **Reconstruction Loss**: Binary cross-entropy between input and reconstructed images
- **KL Divergence**: Regularization term ensuring latent space follows standard normal distribution

```python
loss = reconstruction_loss + kl_divergence
```

## Training

### Training Process

1. **Data Loading**: MNIST dataset is automatically downloaded and preprocessed
2. **Model Initialization**: VAE model is created with specified dimensions
3. **Training Loop**: 
   - Forward pass through encoder-decoder
   - Loss calculation (reconstruction + KL divergence)
   - Backpropagation and parameter updates
   - Progress logging with tqdm

### Checkpointing

- Models are automatically saved every 5 epochs
- Training progress is logged to `runing.log`
- Best model weights are preserved for inference

### Monitoring

The training process provides:
- Real-time loss updates via progress bars
- Periodic logging of training metrics
- Automatic model checkpointing

## Inference

### Generating New Images

After training, generate new handwritten digits:

```python
from train.inference import inference
from utils.data import load_model

# Load trained model
model = load_model(model, "weights/best_model_py_project.pth")

# Generate 5 examples of digit '7'
inference(dataset, model, digit=7, num_examples=5)
```

### Generated Output

- Images are saved in the `generated/` directory
- Format: `generated_{digit}_ex{example_number}.png`
- Each digit (0-9) can be generated with multiple variations

## Results

The trained VAE can:
- **Reconstruct** MNIST digits with high fidelity
- **Generate** new digit samples by sampling from the latent space
- **Interpolate** between different digits in the latent space
- **Learn** meaningful latent representations of handwritten digits

### Expected Performance
- **Reconstruction Quality**: Sharp, recognizable digit reconstructions
- **Generation Diversity**: Varied samples for each digit class
- **Training Stability**: Smooth loss convergence over epochs

## Configuration

### Hardware Requirements
- **CPU**: Any modern multi-core processor
- **GPU**: CUDA-compatible GPU (optional but recommended)
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: ~500MB for dataset and model checkpoints

### Software Dependencies
- Python 3.7+
- PyTorch 1.12.1+
- torchvision 0.13.1+
- Additional packages listed in `requirements.txt`

### Customization Options

**Model Architecture**:
- Adjust hidden dimensions (`--hdim`)
- Modify latent space size (`--zdim`)
- Change activation functions in `VAE.py`

**Training Parameters**:
- Learning rate scheduling
- Different optimizers (Adam, SGD, etc.)
- Custom loss function weights

**Data Pipeline**:
- Support for other datasets
- Data augmentation techniques
- Custom preprocessing transforms

## Acknowledgments

- Original VAE paper: Kingma & Welling (2013)
- PyTorch team for the excellent framework
- MNIST dataset creators and maintainers