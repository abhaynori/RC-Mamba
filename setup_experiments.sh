#!/bin/bash

# Setup script for RC-Mamba vs Transformer comparison experiments

echo "Setting up RC-Mamba experiment environment..."

# Install required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib seaborn pandas numpy tqdm scikit-learn
pip install wandb jupyter notebook

# Create necessary directories
mkdir -p experiments
mkdir -p results
mkdir -p plots

echo "Environment setup complete!"
echo ""
echo "To run the experiments, execute:"
echo "python experiment_framework.py"
echo ""
echo "This will:"
echo "1. Train both Mamba and Transformer models"
echo "2. Generate comparison plots and metrics"
echo "3. Create a detailed analysis for your NeurIPS paper"
