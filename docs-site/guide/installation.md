# Installation

Complete guide to installing NeuroShard on different platforms.

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 2GB | 8GB+ |
| **Storage** | 1GB | 10GB+ |
| **CPU** | 2 cores | 4+ cores |
| **Python** | 3.10 | 3.11 |
| **Network** | 10 Mbps | 100 Mbps+ |

### GPU Support

NeuroShard automatically detects and uses available GPUs:

| GPU | Support Level |
|-----|--------------|
| NVIDIA CUDA | ✅ Full support (recommended) |
| Apple Metal (M1/M2/M3) | ✅ Full support |
| AMD ROCm | ⚠️ Experimental |
| CPU Only | ✅ Supported (slower) |

## Installation Methods

### Method 1: pip (Recommended)

The simplest way to install NeuroShard:

```bash
# Create a virtual environment (recommended)
python -m venv neuroshard-env
source neuroshard-env/bin/activate  # On Windows: neuroshard-env\Scripts\activate

# Install NeuroShard
pip install nexaroa
```

#### With GPU Support (NVIDIA)

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install NeuroShard
pip install nexaroa
```

#### With GPU Support (Apple Silicon)

PyTorch on macOS with Apple Silicon automatically uses Metal:

```bash
pip install torch
pip install nexaroa
```

### Method 2: Docker

Run NeuroShard in a Docker container:

```bash
# Pull the official image
docker pull neuroshard/node:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 9000:9000 \
  -e NEUROSHARD_TOKEN=YOUR_TOKEN \
  neuroshard/node:latest
```

#### Docker Compose

```yaml
version: '3.8'
services:
  neuroshard-node:
    image: neuroshard/node:latest
    ports:
      - "8000:8000"
      - "9000:9000"
    environment:
      - NEUROSHARD_TOKEN=${NEUROSHARD_TOKEN}
    volumes:
      - neuroshard_data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  neuroshard_data:
```

## Verify Installation

After installation, verify everything works:

```bash
# Check version
neuroshard --version
# Output: NeuroShard 0.0.6

# Check available options
neuroshard --help

# Test without running (shows GPU detection)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Platform-Specific Notes

### Windows

1. **Install Python**: Download from [python.org](https://python.org)
2. **Enable Long Paths**: Run as admin:
   ```powershell
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
   ```
3. **Install CUDA Toolkit** (if using NVIDIA GPU): Download from [NVIDIA](https://developer.nvidia.com/cuda-downloads)

### macOS

1. **Install Homebrew**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. **Install Python**: `brew install python@3.11`
3. **For M1/M2/M3**: PyTorch automatically uses Metal Performance Shaders

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# For NVIDIA GPU
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 python3-pip

# For NVIDIA GPU
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
```

## Updating

### pip

```bash
pip install --upgrade nexaroa
```

## Uninstalling

### pip

```bash
pip uninstall nexaroa
```

### Remove Data

```bash
# Remove checkpoints and cache
rm -rf ~/.neuroshard

# On Windows
rd /s /q %USERPROFILE%\.neuroshard
```

## Next Steps

- [Running a Node](/guide/running-a-node) — Configure and start your node
- [Quick Start](/guide/quick-start) — 5-minute setup guide
- [CLI Reference](/guide/cli-reference) — All command options

