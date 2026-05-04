# R1V-A: Vision-Language-Action Agent for Minecraft

A research codebase combining OpenAI VPT (Video Pre-Training), BASALT 2022 competition baseline, and a custom Vision-Language-Action (VLA) agent using LLaVA for multimodal Minecraft gameplay learning.

## 🚀 Quick Start

### Cloud Server Setup
```bash
# Clone repository and run setup
git clone <repository-url> r1v-a
cd r1v-a
chmod +x setup.sh
./setup.sh

# Activate environment
source activate.sh
```

### Local Development
```bash
# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Complete Training Pipeline
```bash
# Run end-to-end training from videos to trained model
python train_pipeline.py \
  --input-videos /path/to/videos \
  --output-dir /path/to/output \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4
```

## 📁 Project Structure

```
r1v-a/
├── train_pipeline.py          # Complete video-to-training pipeline ⭐
├── VLAAgent.py                # Vision-Language-Action agent implementation
├── imitation_learning.py      # VLA agent training script
├── extract_frames.py          # Video frame extraction utilities
├── build_dataset.py           # Dataset building utilities
├── plot_metrics.py            # Training metrics visualization
├── download.py                # Data downloading utilities
├── setup.sh                   # Cloud server setup script ⭐
├── requirements.txt           # Python dependencies
├── test_training_loop.py      # Lightweight training verification
├── VPT/                       # OpenAI VPT implementation
├── basalt-2022-behavioural-cloning-baseline/  # BASALT baseline
├── data/                      # Training datasets
├── trajectories/              # Generated trajectory data
├── logs/                      # Training and execution logs
└── index/                     # Dataset index files
```

## 🤖 Model Architecture

### VLA Agent
- **Backbone**: LLaVA-1.5-7B (frozen)
- **Action Head**: 2-layer MLP (trainable)
- **Input**: RGB frames + text prompts
- **Output**: 23 canonical actions with preserved camera movement

### Action Space
```python
CANONICAL_ACTIONS = [
    "attack", "back", "forward", "jump", "left", "right",
    "sneak", "sprint", "use", "drop", "inventory",
    "hotbar.1" through "hotbar.9", 
    "camera_x", "camera_y", "ESC"
]
```

## 📊 Data Format

### Input Structure
```
videos/
├── task1/
│   ├── video1.mp4           # Gameplay video
│   ├── video1.jsonl         # Frame-by-frame actions
│   ├── video2.mp4
│   └── video2.jsonl
└── task2/
    ├── video3.mp4
    └── video3.jsonl
```

### Processed Structure
```
trajectories/
└── trajectory_task_<name>_length_<N>/
    ├── actions/action_*.jsonl    # Processed actions
    ├── frames/video_*/frame_*.png # Extracted frames
    └── infos/info_*.json         # Task metadata
```

## 🔧 Available Scripts

### Core Training
```bash
# Complete pipeline (recommended)
python train_pipeline.py --input-videos data --output-dir output

# VLA training only
python imitation_learning.py --data-dir trajectories --out-weights model.pt

# VPT behavioral cloning
python VPT/behavioural_cloning.py --data-dir data --out-weights vpt_model.weights
```

### Data Processing
```bash
# Extract frames from videos
python extract_frames.py --data-dir data --workers 8

# Build trajectory dataset
python build_dataset.py input_dir --data_dir trajectories

# Download MineRL datasets
python download.py
```

### Visualization & Testing
```bash
# Plot training metrics
python plot_metrics.py --weights model.pt --save plot.png

# Test training loop
python test_training_loop.py

# System information
python system_info.py  # (created by setup.sh)

# Quick installation test
python quick_test.py   # (created by setup.sh)
```

## ⚙️ Configuration

### Training Parameters
- **Epochs**: 5 (default)
- **Batch Size**: 8 (default)
- **Learning Rate**: 1e-4 (default)
- **Validation Split**: 10% (default)
- **Device**: Auto-detected (CUDA/CPU)

### Model Parameters
- **LLaVA Model**: `llava-hf/llava-1.5-7b-hf` (default)
- **Action Dimensions**: 23 canonical actions
- **Camera Movement**: Preserved X/Y magnitude (not binary)

## 🔍 Key Features

### Training Pipeline (`train_pipeline.py`)
- ✅ Input validation for video/action pairs
- ✅ Parallel frame extraction with progress tracking
- ✅ Automatic trajectory dataset building
- ✅ End-to-end VLA training with validation
- ✅ Comprehensive logging and error handling
- ✅ Training metrics and model checkpointing

### Setup Script (`setup.sh`)
- ✅ Auto-detects and configures GPU/CUDA support
- ✅ Creates Python 3.10 virtual environment
- ✅ Installs all dependencies with compatibility checking
- ✅ Creates utility scripts and documentation
- ✅ Verifies installation with comprehensive tests

### VLA Agent (`VLAAgent.py`)
- ✅ LLaVA backbone integration with frozen weights
- ✅ Mixed precision support (float16 GPU, float32 CPU)
- ✅ Trainable action head with ReLU activation
- ✅ Automatic device and dtype handling

## 📈 Training Metrics

The training process tracks:
- **Train/Validation Loss**: BCEWithLogitsLoss for action prediction
- **Validation Accuracy**: Per-action accuracy metrics
- **Training Progress**: Epoch-by-epoch monitoring

Use `plot_metrics.py` to visualize training curves and model performance.

## 🖥️ Hardware Requirements

### Minimum
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB
- **Python**: 3.10+

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD

## 📝 Dependencies

Core dependencies managed via `requirements.txt`:
- PyTorch 2.0+ (with CUDA support)
- Transformers (HuggingFace)
- OpenCV, Pillow (computer vision)
- NumPy, scikit-learn (data processing)
- tqdm, matplotlib (utilities)

## 🐛 Troubleshooting

### Common Issues

**CUDA not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall PyTorch with CUDA support
```

**Memory issues:**
- Reduce batch size: `--batch-size 4`
- Use CPU training: `--device cpu`

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Dataset format errors:**
- Ensure video/action pairs have matching names
- Check JSONL files are properly formatted
- Verify frame extraction completed successfully

### Getting Help

1. Check logs in `logs/` directory
2. Run `python quick_test.py` to verify installation
3. Use `python system_info.py` for system diagnostics
4. Enable debug logging in training scripts

## 📚 Research Background

This codebase implements research from:
- **VPT**: Video Pre-Training for Minecraft gameplay
- **BASALT**: Human preference learning for Minecraft tasks
- **LLaVA**: Large Language and Vision Assistant

### Citation
```bibtex
@article{r1va2024,
  title={R1V-A: Vision-Language-Action Agent for Minecraft},
  author={Your Name},
  year={2024}
}
```

## 📄 License

This project combines multiple licensed components:
- VPT components: MIT License
- BASALT baseline: MIT License
- Custom VLA agent: MIT License

See individual component directories for specific license details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].