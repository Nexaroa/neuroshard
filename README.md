# NeuroShard AI

**Decentralized AI Training Network**

[![PyPI version](https://badge.fury.io/py/nexaroa.svg)](https://pypi.org/project/nexaroa/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# Install
pip install nexaroa

# Run a node (you'll need a token from neuroshard.com)
neuroshard --token YOUR_TOKEN
```

## What is NeuroShard?

NeuroShard is a decentralized network for training large language models. Contributors share their GPU/CPU power and earn NEURO tokens based on their contribution (Proof of Neural Work).

### Key Features

- **Swarm Architecture** - Fault-tolerant, multipath routing for resilient training
- **Async Training** - DiLoCo-style gradient accumulation for 90%+ bandwidth reduction
- **Earn NEURO** - Get rewarded for contributing compute power
- **Cryptographic Proofs** - ECDSA-signed Proof of Neural Work
- **Web Dashboard** - Monitor your node at `http://localhost:8000`

## Requirements

- Python 3.9+
- 4GB+ RAM (8GB+ recommended)
- GPU optional but recommended for training

### GPU Support

For NVIDIA GPU support, install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Node

```bash
neuroshard --token YOUR_TOKEN
```

### With Custom Port

```bash
neuroshard --token YOUR_TOKEN --port 9000
```

### Connect to Specific Tracker

```bash
neuroshard --token YOUR_TOKEN --tracker tracker.neuroshard.com:8080
```

## Web Dashboard

Once running, open your browser to `http://localhost:8000` to view:

- Node status and role
- Network statistics  
- Training progress
- NEURO balance
- Resource usage

## Architecture

NeuroShard uses a swarm-based architecture for maximum resilience:

- **Dynamic Routing** - If one node fails, work automatically routes to others
- **Activation Buffering** - GPUs never starve waiting for network
- **DiLoCo Training** - Local gradient accumulation reduces sync frequency by 90%+
- **Speculative Checkpoints** - Fast recovery from node failures

## Links

- **Website**: https://neuroshard.com
- **Documentation**: https://docs.neuroshard.com
- **Get a Token**: https://neuroshard.com/register

## License

MIT License - see [LICENSE](LICENSE) for details.
