# Mixture of Experts (MoE)

NeuroShard implements a **Distributed Mixture of Experts** architecture that enables sparse computation across the decentralized network. Unlike traditional MoE implementations where all experts reside on a single machine, NeuroShard distributes experts across nodes, enabling massive model scaling with proportional compute costs.

## Overview

```
                    ┌─────────────────────────────────────────┐
                    │           MoE Transformer Layer          │
                    │                                          │
   Input ──────────►│  ┌──────────────────────────────────┐   │
                    │  │           Self-Attention         │   │
                    │  └──────────────┬───────────────────┘   │
                    │                 │                        │
                    │                 ▼                        │
                    │  ┌──────────────────────────────────┐   │
                    │  │            Router                 │   │ (Always Local)
                    │  │     "Which 2 experts?"           │   │
                    │  └─────────────┬────────────────────┘   │
                    │                │                        │
                    │     ┌──────────┼──────────┐            │
                    │     ▼          ▼          ▼            │
                    │  ┌─────┐   ┌─────┐   ┌─────┐          │
                    │  │ E0  │   │ E3  │   │ E7  │  ...     │ (Local or Remote)
                    │  │Local│   │Peer1│   │Peer2│          │
                    │  └──┬──┘   └──┬──┘   └──┬──┘          │
                    │     │         │         │              │
                    │     └─────────┴─────────┘              │
                    │               │                        │
                    │               ▼                        │
                    │         Weighted Sum ──────────────────► Output
                    └─────────────────────────────────────────┘
```

## Architecture

### Expert Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 8 | Experts per MoE layer |
| `experts_per_token` | 2 | Top-k routing (k experts per token) |
| `target_replicas` | 2 | Redundancy per expert |
| `capacity_factor` | 1.25 | Buffer for load balancing |

### Scaling Properties

- **8 nodes × 8 experts/layer = 64 total experts per layer**
- Total parameters: 8 × 7B = **56B**
- Active parameters per token: (2/8) × 7B ≈ **1.75B**
- Compute scales with k/N (top-k / total experts)

## How It Works

### 1. Router (Always Local)

The router is a simple linear layer that runs on each node:

```python
class MoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=2)  # Top-2
        return weights, indices
```

### 2. Expert Selection

For each token, the router selects the top-k experts (default k=2):

```
Token "hello" → Router → [Expert 3: 0.45, Expert 7: 0.35]
Token "world" → Router → [Expert 1: 0.52, Expert 4: 0.28]
```

### 3. Expert Computation

Experts may be local or remote:

- **Local Expert**: Compute immediately using the local MoE layer
- **Remote Expert**: Forward activations via gRPC to the node holding that expert

```python
if layer_pool.is_expert_local(layer_id, expert_id):
    output = local_experts[expert_id](activations)
else:
    output = grpc_client.ExpertForward(layer_id, expert_id, activations)
```

### 4. Weighted Combination

Expert outputs are combined using router weights:

```python
final_output = sum(weight_i * expert_output_i for i in selected_experts)
```

## Expert Assignment

### Strategy

When a node joins, experts are assigned based on:

1. **Memory capacity**: More memory = more experts
2. **Scarcity**: Prioritize under-replicated experts
3. **Load balancing**: Distribute evenly across layers

```python
def assign_experts_to_node(node_id, layer_ids, memory_mb):
    for layer_id in layer_ids:
        # Find under-replicated experts
        expert_counts = get_expert_replica_counts(layer_id)
        
        # Assign up to max_experts based on memory
        max_experts = min(8, memory_mb * 0.3 / expert_memory_mb)
        
        # Prioritize scarcest experts
        for expert_id in sorted_by_scarcity(expert_counts):
            if count[expert_id] < target_replicas:
                assign(node_id, layer_id, expert_id)
```

### Scarcity Bonus

Under-replicated experts earn bonus NEURO to incentivize balanced distribution:

```
scarcity_bonus = 1.0 + 0.5 × (target_replicas - current_replicas) / target_replicas
```

| Current Replicas | Target | Scarcity Bonus |
|------------------|--------|----------------|
| 0 | 2 | 1.50× |
| 1 | 2 | 1.25× |
| 2+ | 2 | 1.00× |

## Load Balancing

### Auxiliary Losses

To prevent routing collapse (all tokens going to few experts), MoE uses auxiliary losses:

**1. Load Balancing Loss**

Encourages uniform expert utilization:

```
L_aux = α × N × Σ(tokens_fraction_i × prob_fraction_i)
```

Where:
- `α = 0.01` (coefficient)
- `N = num_experts`
- `tokens_fraction_i` = fraction of tokens routed to expert i
- `prob_fraction_i` = average router probability for expert i

**2. Router Z-Loss**

Prevents router logits from growing too large:

```
L_z = β × mean(log_sum_exp(logits)²)
```

Where `β = 0.001`.

### Capacity Limiting

Each expert has a capacity limit to prevent overload:

```python
capacity = (batch_size × seq_len × top_k / num_experts) × capacity_factor
```

Tokens exceeding capacity are dropped (during training) or queued (during inference).

## gRPC Interface

### ExpertForward

Forward activations through a remote expert:

```protobuf
message ExpertForwardRequest {
    string request_id = 1;
    int32 layer_id = 2;
    int32 expert_id = 3;
    bytes activations = 4;
    repeated int64 shape = 5;
    string dtype = 6;
}

message ExpertForwardResponse {
    bool success = 1;
    bytes output = 2;
    float utilization = 3;  // Current load 0-1
}
```

### GetExpertStatus

Query expert availability for routing:

```protobuf
message ExpertStatusRequest {
    int32 layer_id = 1;
    int32 expert_id = 2;
}

message ExpertStatusResponse {
    bool available = 1;
    float utilization = 2;
    int32 replicas = 3;
}
```

### AnnounceExpert

Announce expert availability to network:

```protobuf
message AnnounceExpertRequest {
    string node_id = 1;
    int32 layer_id = 2;
    int32 expert_id = 3;
    float capacity = 4;
}

message AnnounceExpertResponse {
    bool accepted = 1;
    int32 total_replicas = 2;
    float scarcity_bonus = 3;
}
```

## Performance Considerations

### Latency

| Scenario | Latency |
|----------|---------|
| Both experts local | ~5ms |
| 1 local + 1 remote | ~15ms |
| Both experts remote | ~25ms |

### Optimizations

1. **Prefetch popular experts**: Cache commonly-used expert weights
2. **Async remote calls**: Don't block on remote expert responses
3. **Batch remote requests**: Combine multiple tokens to same expert
4. **Expert replication**: High-demand experts get more replicas

## Integration with PoNW

MoE expert contributions are tracked in Proof of Neural Work:

```python
@dataclass
class PoNWProof:
    # ... other fields ...
    moe_expert_count: int = 0     # Experts held by this node
    moe_scarcity_avg: float = 1.0 # Average scarcity bonus
```

Final reward calculation:

```python
total_reward = base_reward × stake_multiplier × role_multiplier × moe_scarcity_bonus
```

## Example: Node with MoE

```
$ neuroshard --device cuda --memory 8000

[INFO] Registered node with 8000MB capacity
[INFO] Assigned layers: [0, 1, 2, 3]
[MoE] Layer 0: Experts [0, 1, 2, 3] (4 local)
[MoE] Layer 1: Experts [0, 1] (2 local)
[MoE] Layer 2: Experts [4, 5, 6, 7] (4 local)
[MoE] Layer 3: Experts [2, 3] (2 local)
[INFO] Total: 12 experts across 4 layers
[INFO] Training reward: 1.35× (avg scarcity bonus)
```

## Future Work

1. **Expert routing optimization**: ML-based routing to minimize cross-node calls
2. **Dynamic expert migration**: Move experts based on utilization patterns
3. **Hierarchical MoE**: Nested experts for multi-level specialization
4. **Expert distillation**: Compress multiple experts into single dense expert

---

*MoE enables NeuroShard to scale to frontier model sizes while keeping compute costs proportional to the active parameter count.*

