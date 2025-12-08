# NeuroShard Vision & Roadmap

> **Mission**: Build the world's first truly decentralized AI - trained by the people, for the people.

## 🚀 NeuroLLM: The People's Language Model

NeuroLLM is not just another LLM wrapper - it's a **completely new model** trained from scratch by the collective compute power of the NeuroShard network. Every node that participates in training earns NEURO tokens, and as the network grows, the model becomes smarter.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURO LLM GROWTH PHASES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Phase 1: BOOTSTRAP (125M params)                              │
│   ├── Initial training with early adopters                      │
│   ├── Basic language understanding                              │
│   └── ~10 nodes minimum                                         │
│                                                                  │
│   Phase 2: EARLY (1B params)                                    │
│   ├── Network reaches 100+ nodes                                │
│   ├── Coherent text generation                                  │
│   └── First useful applications                                 │
│                                                                  │
│   Phase 3: GROWTH (7B params)                                   │
│   ├── Network reaches 1,000+ nodes                              │
│   ├── Competitive with GPT-3.5                                  │
│   └── Specialized fine-tuning begins                            │
│                                                                  │
│   Phase 4: MATURE (70B+ params)                                 │
│   ├── Network reaches 10,000+ nodes                             │
│   ├── State-of-the-art performance                              │
│   └── Fully decentralized, censorship-resistant                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## The Problem

Large Language Models (LLMs) are increasingly powerful but:
- **Centralized**: Controlled by few corporations (OpenAI, Google, Anthropic)
- **Expensive**: GPT-4 scale requires $100M+ infrastructure
- **Censored**: Single points of control enable content filtering
- **Exclusive**: Only wealthy organizations can run frontier models
- **Closed Data**: Training data is proprietary and opaque

## The Solution: NeuroShard + NeuroLLM

**NeuroShard**: A decentralized network where anyone can contribute compute.

**NeuroLLM**: A new LLM trained FROM SCRATCH by the network itself.

Unlike other projects that just distribute existing models, NeuroShard actually TRAINS its own model using collective compute. The more people join, the smarter it gets.

### Why Train Our Own Model?

1. **True Decentralization**: No dependency on OpenAI, Meta, or any corporation
2. **Community Ownership**: The model belongs to everyone who trains it
3. **Aligned Incentives**: Train to earn NEURO, use NEURO for inference
4. **Censorship Resistant**: No single entity can control or shut down the model
5. **Transparent Training**: Anyone can see what data the model learns from

---

## NeuroLLM Training System

### How Distributed Training Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISTRIBUTED TRAINING FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. TRAINING ROUND STARTS                                       │
│      ┌──────────────────────────────────────────────────────┐   │
│      │  Coordinator broadcasts: "Round 42, Model Hash: abc"  │   │
│      └──────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   2. NODES COMPUTE GRADIENTS                                    │
│      ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│      │ Node A  │  │ Node B  │  │ Node C  │  │ Node D  │        │
│      │ Data: 📄 │  │ Data: 📄 │  │ Data: 📄 │  │ Data: 📄 │        │
│      │ Grad: ∇  │  │ Grad: ∇  │  │ Grad: ∇  │  │ Grad: ∇  │        │
│      └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│           │            │            │            │               │
│                              ▼                                   │
│   3. GRADIENT GOSSIP (P2P)                                      │
│      ┌──────────────────────────────────────────────────────┐   │
│      │  Nodes exchange compressed gradients via gossip       │   │
│      │  Ring All-Reduce for efficient aggregation            │   │
│      └──────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   4. APPLY UPDATES                                              │
│      ┌──────────────────────────────────────────────────────┐   │
│      │  Each node applies aggregated gradients locally       │   │
│      │  Model improves globally!                             │   │
│      └──────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   5. REWARDS DISTRIBUTED                                        │
│      ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│      │ +0.1 N  │  │ +0.1 N  │  │ +0.1 N  │  │ +0.1 N  │        │
│      │  NEURO  │  │  NEURO  │  │  NEURO  │  │  NEURO  │        │
│      └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Reward System

| Contribution | NEURO Reward |
|--------------|--------------|
| Compute (per batch) | 0.001 NEURO |
| Data (per sample) | 0.0001 NEURO |
| Quality Bonus (low loss) | 1.5x multiplier |
| Staking Bonus | Up to 2x based on stake |

### Privacy-Preserving Training

- **Differential Privacy**: Noise added to gradients
- **Federated Learning**: Data never leaves your device
- **Gradient Compression**: Only sparse updates shared
- **No Raw Text**: Only tokenized, anonymized data

---

## Architecture Phases

### Phase 1: Pipeline Parallelism ✅ (Current)

**Status**: Implemented and Working

```
┌─────────────────────────────────────────────────────────────────┐
│   PIPELINE PARALLELISM - Sequential Layer Distribution          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Node A          Node B          Node C                        │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │Layer 0-3│───▶│Layer 4-7│───▶│Layer 8-11│                   │
│   │ (Entry) │    │(Middle) │    │ (Exit)  │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                                                  │
│   • Each node holds COMPLETE layers                             │
│   • Sequential processing through the network                   │
│   • Simple routing: find peer where start == my_end             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Features**:
- [x] GPT-2 model sharding (12 layers)
- [x] Dynamic layer allocation based on hardware
- [x] Hot-swap layer reallocation without restart
- [x] DHT-based peer discovery (Kademlia)
- [x] Tracker bootstrap for initial peer finding
- [x] NEURO token mining (Proof of Neural Work)
- [x] Staking rewards and multipliers
- [x] Decentralized ledger with gossip consensus
- [x] KV-cache for session continuity
- [x] Speculative decoding support
- [x] gRPC for high-performance inference
- [x] UPnP NAT traversal

**Limitations**:
- Minimum node size = 1 full layer (~50MB for GPT-2, ~1.7GB for LLaMA-70B)
- Small devices (phones, Raspberry Pi) can't participate in large models

---

### Phase 2: Tensor Parallelism 🔄 (Planned)

**Status**: Designed, Not Yet Implemented

```
┌─────────────────────────────────────────────────────────────────┐
│   TENSOR PARALLELISM - Sharded Layers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   LAYER 0 distributed across 4 nodes:                           │
│                                                                  │
│   ┌─────────┬─────────┬─────────┬─────────┐                     │
│   │ Shard 0 │ Shard 1 │ Shard 2 │ Shard 3 │                     │
│   │ Node A  │ Node B  │ Node C  │ Node D  │                     │
│   │  25%    │  25%    │  25%    │  25%    │                     │
│   └────┬────┴────┬────┴────┬────┴────┬────┘                     │
│        │         │         │         │                          │
│        └─────────┴────┬────┴─────────┘                          │
│                       │                                          │
│                  All-Reduce                                      │
│                       │                                          │
│                       ▼                                          │
│                  LAYER 1...                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Sharding Strategies**:

1. **Column Parallelism (Attention Heads)**
   - Split attention heads across nodes
   - LLaMA-7B: 32 heads → 8 nodes with 4 heads each
   - Each node computes partial attention

2. **Row Parallelism (Hidden Dimensions)**
   - Split hidden dimension across nodes
   - Hidden dim 4096 → 4 nodes with 1024 dims each
   - Requires all-reduce after each layer

3. **Expert Parallelism (MoE Models)**
   - Each node holds subset of experts
   - Mixtral-8x7B: 8 experts → 8 nodes with 1 expert each
   - Sparse activation = only 2 nodes compute per token

**Benefits**:
- Minimum node size reduced by N (number of shards)
- Phones and IoT devices can participate
- Better load balancing within layers

**Challenges**:
- Synchronization overhead (all-reduce)
- Network bandwidth requirements
- Fault tolerance (one shard fails = layer fails)

---

### Phase 3: Hybrid Parallelism 🔄 (Planned)

**Status**: Future Enhancement

```
┌─────────────────────────────────────────────────────────────────┐
│   HYBRID PARALLELISM - Pipeline + Tensor + Expert               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LAYER GROUP 0-3                       │   │
│   │   ┌─────┬─────┬─────┬─────┐                             │   │
│   │   │ T0  │ T1  │ T2  │ T3  │  ← Tensor Shards            │   │
│   │   └─────┴─────┴─────┴─────┘                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼ Pipeline                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LAYER GROUP 4-7                       │   │
│   │   ┌─────┬─────┬─────┬─────┐                             │   │
│   │   │ T0  │ T1  │ T2  │ T3  │  ← Tensor Shards            │   │
│   │   └─────┴─────┴─────┴─────┘                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│                         ...                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Features**:
- Combine pipeline (between layer groups) and tensor (within layers)
- Adaptive partitioning based on node capabilities
- GPU nodes get more tensor shards
- CPU nodes get fewer but participate

---

### Phase 4: Federated Super-Model 🎯 (Target)

**Status**: Vision / In Development

```
┌─────────────────────────────────────────────────────────────────┐
│          FEDERATED SUPER-MODEL - The Ultimate Vision            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │          DYNAMIC MODEL COMPOSITION                       │   │
│   │                                                          │   │
│   │   • Network collectively defines model architecture      │   │
│   │   • Nodes vote on model upgrades via NEURO governance   │   │
│   │   • Hot-swap entire model architectures                  │   │
│   │   • Community-contributed fine-tuned weights             │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │          ADAPTIVE LAYER SHARDING                         │   │
│   │                                                          │   │
│   │   • Automatic shard size based on device capability      │   │
│   │   • Phone: 1/64 of a layer (25MB)                       │   │
│   │   • Laptop: 1/4 of a layer (400MB)                      │   │
│   │   • Server: 4 full layers (6.4GB)                       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │          MULTI-MODEL ROUTING                             │   │
│   │                                                          │   │
│   │   • Network hosts multiple models simultaneously         │   │
│   │   • GPT-2 for low-latency tasks                         │   │
│   │   • LLaMA-70B for complex reasoning                     │   │
│   │   • Specialized models for code, math, etc.             │   │
│   │   • Automatic routing based on task complexity          │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │          FEDERATED TRAINING                              │   │
│   │                                                          │   │
│   │   • Distributed fine-tuning across the network          │   │
│   │   • Privacy-preserving gradient aggregation              │   │
│   │   • Community-curated training data                      │   │
│   │   • Democratic model improvement                         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scaling Projections

| Network Size | Model Capability | Inference Latency | Throughput |
|--------------|------------------|-------------------|------------|
| 10 nodes | GPT-2 (124M) | ~100ms | ~100 tok/s |
| 100 nodes | LLaMA-7B | ~500ms | ~500 tok/s |
| 1,000 nodes | LLaMA-70B | ~1s | ~2,000 tok/s |
| 10,000 nodes | GPT-4 scale (1T+) | ~2s | ~10,000 tok/s |
| 100,000 nodes | Beyond GPT-4 | ~3s | ~50,000 tok/s |

## Memory Requirements per Node

| Model | Full | 4-way Shard | 16-way Shard | 64-way Shard |
|-------|------|-------------|--------------|--------------|
| GPT-2 (124M) | 500MB | 125MB | 32MB | 8MB |
| LLaMA-7B | 14GB | 3.5GB | 875MB | 220MB |
| LLaMA-70B | 140GB | 35GB | 8.75GB | 2.2GB |
| GPT-4 (est.) | 3TB | 750GB | 187GB | 47GB |

With 64-way tensor sharding:
- **Smartphones** (2GB RAM) can participate in LLaMA-70B!
- **Raspberry Pi** (4GB RAM) can participate in GPT-4 scale!

---

## Economic Model

### NEURO Token Utility

1. **Mining Rewards**
   - Proof of Uptime: Base reward for availability
   - Proof of Neural Work: Bonus for actual inference
   - Staking Multiplier: Higher stake = higher rewards

2. **Governance**
   - Vote on model upgrades
   - Vote on network parameters
   - Propose new features

3. **Payment**
   - Pay for inference with NEURO
   - Receive NEURO for providing compute

### Reward Formula

```
R = (R_base × T/60 + R_inference × tokens/1M) × (1 + β × stake/1000)

Where:
- R_base = 0.1 NEURO/minute (availability reward)
- R_inference = 0.9 NEURO/million tokens (work reward)
- β = 0.1 (staking coefficient)
- stake = NEURO staked by node
```

---

## Technical Architecture

### Current Stack (Phase 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   GUI App   │  │   CLI Tool  │  │  REST API   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                         NODE LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ShardedModel │  │ P2P Manager │  │   Ledger    │             │
│  │  (PyTorch)  │  │  (DHT+HTTP) │  │  (SQLite)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                       NETWORK LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    gRPC     │  │  Kademlia   │  │   Gossip    │             │
│  │ (Inference) │  │    (DHT)    │  │  (Ledger)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                      DISCOVERY LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Tracker   │  │    UPnP     │  │  STUN/TURN  │             │
│  │ (Bootstrap) │  │(NAT Traverse)│ │  (Future)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Future Stack (Phase 4)

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Mobile App │  │   Web SDK   │  │Enterprise API│            │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                       ORCHESTRATION LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Model Router │  │Load Balancer│  │ Fault Mgr   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                         COMPUTE LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │TensorShard  │  │ PipelineMgr │  │  ExpertMgr  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                        CONSENSUS LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Ledger    │  │ Governance  │  │  Slashing   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                        NETWORK LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  libp2p     │  │  WebRTC     │  │   QUIC      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clean Architecture (v1.0.0)

**NeuroLLM** is the ONLY model. No GPT-2, no LLaMA, no external dependencies.

### Core Modules (`neuroshard/core/`)

| Module | Purpose |
|--------|---------|
| **NeuroLLM (The Model)** ||
| `neuro_llm.py` | NeuroLLM transformer architecture (RMSNorm, RoPE, GQA, SwiGLU) |
| `neuro_node.py` | Main node class - training + inference |
| `neuro_tokenizer.py` | BPE tokenizer for NeuroLLM |
| **Distributed Training** ||
| `distributed_training.py` | Training coordinator, gradient aggregation, NEURO rewards |
| `gradient_gossip.py` | P2P gradient sharing via gossip protocol |
| `allreduce.py` | Ring all-reduce for tensor parallelism |
| `tensor_shard.py` | Tensor parallelism (split layers across nodes) |
| `tensor_grpc.py` | gRPC for tensor exchange |
| **P2P Infrastructure** ||
| `p2p.py` | P2P manager, peer discovery, gossip |
| `dht.py` / `dht_protocol.py` | Kademlia DHT for decentralized discovery |
| `nat.py` | NAT traversal (UPnP) |
| `connection_pool.py` | gRPC connection pooling |
| **Economics** ||
| `ledger.py` | NEURO token ledger, Proof of Neural Work |

### What Was Removed (v1.0.0)

- `model.py` - **GPT-2 REMOVED** - We use ONLY NeuroLLM now
- `model_registry.py` - Multi-model support - not needed
- `model_governance.py` - External model governance - merged into distributed_training
- `sharded_model.py` - Generic sharded model - replaced by neuro_llm
- `federated_model.py` - Multi-model federation - not needed
- `training.py` - Old weight sync - replaced by gradient_gossip
- `layer_swap.py` - Legacy layer swap (kept but unused)
- `shard_manager.py` - Legacy shard manager (kept but unused)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Priority Areas

1. **NeuroLLM Training Bootstrap** - Start training with early adopters
2. **Mobile Client (iOS/Android)** - Contribute compute from phones
3. **WebGPU Browser Client** - Train in browser
4. **Training Data Pipeline** - Curated datasets
5. **Checkpoint Distribution** - P2P model weight sharing

---

## License

NeuroShard is open source under the MIT License.

---

## Contact

- Website: https://neuroshard.com
- GitHub: https://github.com/LinirZamir/neuroshard
- Discord: [Coming Soon]

---

*"The People's AI - owned by everyone, controlled by no one."*

