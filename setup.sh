#!/bin/bash
# IP-Adapter Training Kit Setup Script
# This script helps you get started quickly with the toolkit

set -e

echo "🚀 IP-Adapter Training Kit Setup"
echo "=================================="
echo

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "   Python version: $python_version"

# Check if version is 3.9+
required_version="3.9"
if [[ $(echo "$python_version" | cut -d'.' -f1,2) < $required_version ]]; then
    echo "❌ Error: Python 3.9+ required (found $python_version)"
    exit 1
fi
echo "   ✅ Python version OK"
echo

# Check for GPU
echo "🖥️  Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   🎮 NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
elif [[ $(uname) == "Darwin" ]]; then
    echo "   🍎 Apple Silicon Mac detected (MPS support)"
else
    echo "   ⚠️  No GPU detected - training will be slow on CPU"
fi
echo

# Create virtual environment
echo "🐍 Setting up virtual environment..."
if [[ ! -d "venv" ]]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "   Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip
echo "   ✅ pip upgraded"
echo

# Install requirements
echo "🔧 Installing requirements..."
if [[ -f "requirements/requirements.txt" ]]; then
    echo "   Installing core dependencies..."
    pip install -r requirements/requirements.txt
    echo "   ✅ Core dependencies installed"
else
    echo "   Installing basic dependencies..."
    pip install torch torchvision diffusers transformers accelerate safetensors pillow
    echo "   ✅ Basic dependencies installed"
fi
echo

# Setup IP-Adapter repository
echo "📥 Setting up IP-Adapter repository..."
if [[ ! -d "IP-Adapter-main" ]]; then
    echo "   Downloading IP-Adapter..."
    git clone https://github.com/tencent-ailab/IP-Adapter.git IP-Adapter-main
    echo "   ✅ IP-Adapter downloaded"
else
    echo "   ✅ IP-Adapter already exists"
fi
echo

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p generated_images
mkdir -p logs
mkdir -p checkpoints
echo "   ✅ Directories created"
echo

# Run a quick test
echo "🧪 Running quick test..."
python3 -c "
import torch
import diffusers
import transformers
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Diffusers version: {diffusers.__version__}')
print(f'Transformers version: {transformers.__version__}')

# Check device
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) available')
else:
    print('Using CPU (will be slower)')
"
echo

# Show next steps
echo "🎉 Setup Complete!"
echo "=================="
echo
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Train on mini dataset: python scripts/train_mini_dataset.py"
echo "3. Convert checkpoint: python scripts/convert_checkpoint.py" 
echo "4. Generate images: python scripts/use_trained_model_proper.py --test"
echo
echo "📚 Documentation:"
echo "- Quick start: README.md"
echo "- Training guide: docs/MINI_DATASET_TRAINING.md"
echo "- Dataset guide: docs/CUSTOM_DATASET_GUIDE.md"
echo
echo "Happy training! 🎨✨" 