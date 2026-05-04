#!/bin/bash
# Download LLaVA model for offline cluster training
# Run this on JURECA DC LOGIN NODE (with internet access)

set -e  # Exit on any error

echo "🚀 LLaVA Model Download Script for JURECA DC"
echo "============================================="
echo "This script must be run on the LOGIN NODE (with internet access)"
echo "Date: $(date)"
echo "User: $USER"
echo "Host: $(hostname)"
echo ""

# Check if we're on login node (basic check)
if [[ $(hostname) != *"login"* ]] && [[ $(hostname) != *"jureca"* ]]; then
    echo "⚠️  Warning: This doesn't look like a login node"
    echo "   Make sure you have internet access before continuing"
    echo ""
fi

# Load required modules for Python 3.10
echo "📦 Loading required modules..."
module load Stages/2023
module load GCC/11.3.0
module load Python/3.10.4
module load CUDA/11.7

echo "✅ Modules loaded"
echo "Python: $(python --version)"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first:"
    echo "   python3.10 -m venv .venv"
    echo "   source .venv/bin/activate" 
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Verify Python and packages
echo "✅ Environment activated"
echo "Python path: $(which python)"
echo "Pip packages:"
pip list | grep -E "(torch|transformers)" || echo "⚠️  Required packages not found"
echo ""

# Check internet connectivity
echo "🌐 Testing internet connectivity..."
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    echo "✅ Internet connection OK"
else
    echo "❌ No internet connection!"
    echo "This script must run on a login node with internet access"
    exit 1
fi
echo ""

# Check disk space
echo "💾 Checking disk space..."
df -h . | head -2
echo ""

AVAILABLE=$(df . | tail -1 | awk '{print $4}')
if [ $AVAILABLE -lt 15728640 ]; then  # 15 GB in KB
    echo "⚠️  Warning: Less than 15 GB available space"
    echo "   LLaVA model requires ~13 GB"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Create cache directory
echo "📁 Setting up cache directory..."
mkdir -p hf_cache
export HF_HOME="$(pwd)/hf_cache"
export TRANSFORMERS_CACHE="$(pwd)/hf_cache/transformers"

echo "Cache directory: $HF_HOME"
echo ""

# Run Python download script
echo "🐍 Starting Python download script..."
echo "This will take 10-15 minutes for ~13 GB download"
echo ""

python download_llava.py

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! LLaVA model downloaded successfully"
    echo ""
    echo "📊 Cache information:"
    echo "Size: $(du -sh hf_cache | cut -f1)"
    echo "Location: $(pwd)/hf_cache"
    echo ""
    echo "📋 Next steps:"
    echo "1. Your model is ready for offline training"
    echo "2. Submit your training job: sbatch slurm_train.sh"
    echo "3. The job will automatically use the cached model"
    echo ""
    echo "🔧 Environment setup for manual use:"
    echo "   source set_hf_cache.sh"
    echo ""
else
    echo ""
    echo "❌ FAILED! Download unsuccessful"
    echo "Check the error messages above"
    echo ""
    echo "🛠️  Troubleshooting:"
    echo "- Verify internet connection: curl https://huggingface.co"
    echo "- Check Python packages: pip list | grep transformers"
    echo "- Try manual installation: pip install --upgrade transformers torch"
    echo ""
    exit 1
fi