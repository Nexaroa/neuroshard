# NeuroShard Installation Guide

## Quick Installation

NeuroShard is installed via pip and works on Windows, macOS, and Linux.

### Basic Installation

```bash
# Install NeuroShard
pip install nexaroa

# Run your node
neuroshard --token YOUR_WALLET_TOKEN
```

The dashboard will automatically open at `http://localhost:8000`.

---

## Platform-Specific Setup

### 🪟 Windows

#### CPU Version (Default)
**Recommended for:**
- Intel/AMD processors without NVIDIA GPU
- Laptops without dedicated graphics
- Testing/light usage

**Installation:**
```bash
pip install nexaroa
neuroshard --token YOUR_WALLET_TOKEN
```

**Performance:**
- Training: ~10s per step
- Layers: 40-60 (depending on RAM)
- Memory: Low overhead

---

#### NVIDIA GPU Version
**Recommended for:**
- NVIDIA GPUs (GTX 1060+, RTX series)
- Serious training nodes
- Maximum performance

**Installation:**
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install NeuroShard
pip install nexaroa

# Run your node
neuroshard --token YOUR_WALLET_TOKEN
```

**Requirements:**
- NVIDIA GPU with 4GB+ VRAM
- CUDA-compatible drivers (version 450+)

**Performance:**
- Training: ~2s per step (5-10x faster!)
- Layers: 100-150+ (GPU memory is separate)
- Memory: Offloads compute to GPU

**Check your GPU:** Open CMD and run `nvidia-smi`

---

### 🍎 macOS

**Installation:**
```bash
pip install nexaroa
neuroshard --token YOUR_WALLET_TOKEN
```

**Features:**
- **Intel Macs:** CPU training
- **M1/M2/M3/M4 Macs:** Automatic GPU acceleration via Metal (MPS)
- One installation works for all Macs!

**Performance:**
- Intel: ~10s per step (CPU)
- M1/M2/M3/M4: ~3s per step (GPU via MPS)

---

### 🐧 Linux

**Installation:**
```bash
pip install nexaroa
neuroshard --token YOUR_WALLET_TOKEN
```

**For NVIDIA GPU:**
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install NeuroShard
pip install nexaroa

# Run your node
neuroshard --token YOUR_WALLET_TOKEN
```

---

## 🎯 Which Version Should You Use?

### Quick Decision Tree:

```
Do you have an NVIDIA GPU?
├─ YES → Install PyTorch with CUDA, then nexaroa
│         ├─ Works? → GPU acceleration enabled
│         └─ Fails? → Use CPU version (just nexaroa)
│
├─ NO → What's your system?
│       ├─ Mac M1/M2/M3/M4 → Install nexaroa (auto GPU via MPS)
│       ├─ Windows/Intel/Linux → Install nexaroa (CPU)
│       └─ All platforms use the same package!
```

---

## 📊 Performance Comparison

| Hardware | Version | Training Speed | Recommended Layers |
|----------|---------|----------------|-------------------|
| Intel CPU (8 cores) | CPU | 10s/step | 40-60 layers |
| AMD CPU (16 cores) | CPU | 6s/step | 80-100 layers |
| NVIDIA RTX 3060 (6GB) | GPU | 2s/step | 120-150 layers |
| NVIDIA RTX 4090 (24GB) | GPU | 0.5s/step | 400+ layers |
| Apple M1 | Mac (MPS) | 4s/step | 60-80 layers |
| Apple M2 Pro | Mac (MPS) | 2.5s/step | 100-120 layers |

---

## 🔧 Troubleshooting

### "GPU version not working"
1. Verify your GPU: `nvidia-smi` (Windows/Linux)
2. Update drivers: [NVIDIA Downloads](https://nvidia.com/drivers)
3. Minimum: Driver 450+, CUDA 11.0+
4. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "Package not found"
- Ensure pip is up to date: `pip install --upgrade pip`
- Check PyPI: [pypi.org/project/nexaroa](https://pypi.org/project/nexaroa)

### "Command not found"
- After installation, ensure `~/.local/bin` is in your PATH (Linux/Mac)
- Or use: `python -m neuroshard --token YOUR_TOKEN`

---

## 💾 Storage Requirements

### Application Size:
- **Package:** ~50MB (pip install)
- **Dependencies:** ~500MB-2GB (PyTorch, etc.)

### Runtime Data:
- **Training shards:** 100MB - 10GB (configurable in Settings)
- **Model checkpoints:** 500MB - 2GB (your trained weights)
- **Logs:** 5-50MB

**Total:** 1-15GB depending on settings

---

## 🚀 First-Time Setup

1. **Install** NeuroShard: `pip install nexaroa`
2. **Get your Node Token** from [neuroshard.com/register](https://neuroshard.com/register)
3. **Run** your node: `neuroshard --token YOUR_WALLET_TOKEN`
4. **Dashboard** opens automatically at `http://localhost:8000`
5. **Set RAM/Storage limits** in Settings (optional)
6. **Start earning NEURO!**

---

## ❓ FAQ

### Can I switch from CPU to GPU version later?
**Yes!** Your training progress is saved in `~/.neuroshard/` and works with both versions. Just reinstall PyTorch with CUDA support.

### Which version earns more NEURO?
**GPU version** - faster training = more steps = more rewards. Roughly 5-10x more NEURO/day.

### Can I run multiple nodes?
**Yes!** Each node needs a different port. Use `--port` flag: `neuroshard --token TOKEN1 --port 8000` and `neuroshard --token TOKEN2 --port 8001`

### Do I need to download the whole model?
**No!** The model is distributed across the network. You only hold layers based on your RAM.

### Where is my data stored?
All data is stored in `~/.neuroshard/` (or `%USERPROFILE%\.neuroshard\` on Windows):
- `checkpoints/` - Model weights
- `data_cache/` - Training data
- `logs/` - Node logs

---

## 📞 Support

- **Discord:** [discord.gg/4R49xpj7vn](https://discord.gg/4R49xpj7vn)
- **Documentation:** [docs.neuroshard.com](https://docs.neuroshard.com)
- **Website:** [neuroshard.com](https://neuroshard.com)
