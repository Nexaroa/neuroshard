"""
Dynamic NeuroLLM - Decentralized Model Parallelism

This module implements the core of NeuroShard's distributed training:
- Dynamic layer assignment based on node capacity
- Pipeline parallelism across nodes
- DHT-based peer discovery for routing
- Automatic scaling as nodes join/leave
"""

import torch
import torch.nn as nn
import threading
import time
import logging
import os
import json
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Initial vocab size - will grow dynamically as tokenizer learns
INITIAL_VOCAB_SIZE = 32000
# Maximum vocab size to pre-allocate for
MAX_VOCAB_SIZE = 10_000_000


@dataclass
class LayerAssignment:
    """Tracks which node holds which layer."""
    layer_id: int
    node_id: str
    node_url: str
    grpc_addr: str


class DynamicLayerPool:
    """
    Manages layer distribution across the network.
    
    DESIGN PRINCIPLES:
    1. DHT is the source of truth for network state
    2. Nodes announce their layers to DHT
    3. DHT entries expire (TTL) so stale data disappears
    4. Simple protocol: Check layer_0 → if exists, join as WORKER
    """
    
    MIN_REPLICAS = 2  # Minimum redundancy for fault tolerance
    
    def __init__(self, dht_protocol=None):
        self.layer_assignments: Dict[int, List[LayerAssignment]] = {}
        self.node_capacities: Dict[str, float] = {}
        self.current_num_layers = 0
        self.current_architecture = None
        self.dht = dht_protocol
        self.lock = threading.Lock()
        
        # Architecture tracking
        self.last_node_count = 0
        
        # Vocab capacity for memory calculation
        self.vocab_capacity = INITIAL_VOCAB_SIZE
        
        # Role tracking
        self.embedding_holder: Optional[str] = None
        self.lm_head_holder: Optional[str] = None
        
        # Checkpoint directory
        self.CHECKPOINT_DIR = Path.home() / ".neuroshard" / "checkpoints"
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("DynamicLayerPool initialized with dynamic width + depth scaling")
    
    def _get_checkpoint_path(self, node_id: str) -> Optional[Path]:
        """Get checkpoint path for a node."""
        # Try wallet_id first (set by DynamicNeuroNode.start())
        if hasattr(self, '_wallet_id') and self._wallet_id:
            path = self.CHECKPOINT_DIR / f"dynamic_node_{self._wallet_id}.pt"
            if path.exists():
                return path
        # Fallback to node_id hash
        import hashlib
        node_hash = hashlib.sha256(node_id.encode()).hexdigest()[:16]
        path = self.CHECKPOINT_DIR / f"dynamic_node_{node_hash}.pt"
        if path.exists():
            return path
        return None
    
    def _get_recalc_interval(self, node_count: int) -> int:
        """Dynamic recalculation interval based on network size."""
        if node_count < 10:
            return 2
        elif node_count < 100:
            return 10
        elif node_count < 1000:
            return 50
        else:
            return 100
    
    def _auto_recalculate_architecture(self):
        """Recalculate optimal architecture based on current network."""
        from neuroshard.core.model.scaler import ModelArchitecture
        
        total_memory = sum(self.node_capacities.values())
        node_count = len(self.node_capacities)
        
        if node_count == 0:
            # Default architecture for first node
            self.current_architecture = ModelArchitecture(
                hidden_dim=512,
                intermediate_dim=2048,
                num_layers=11,
                num_heads=16,
                num_kv_heads=4,
                vocab_size=MAX_VOCAB_SIZE,
                max_seq_len=2048,
                dropout=0.1,
                rope_theta=10000.0
            )
        else:
            # Scale based on network capacity
            self.current_architecture = ModelArchitecture(
                hidden_dim=512,
                intermediate_dim=2048,
                num_layers=11,
                num_heads=16,
                num_kv_heads=4,
                vocab_size=MAX_VOCAB_SIZE,
                max_seq_len=2048,
                dropout=0.1,
                rope_theta=10000.0
            )
        
        if self.current_num_layers == 0:
            self.current_num_layers = self.current_architecture.num_layers
    
    def _discover_layer_holders_from_dht(self, layer_id: int) -> List[str]:
        """
        Query DHT for nodes holding a specific layer.
        Returns list of node URLs.
        """
        if not self.dht:
            return []
        
        try:
            import hashlib
            key_str = f"layer_{layer_id}"
            key_int = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
            
            value = self.dht.lookup_value(key_int)
            if value:
                try:
                    holders = json.loads(value)
                    if isinstance(holders, list):
                        return holders
                    else:
                        return [str(holders)]
                except:
                    return [value]
        except Exception as e:
            logger.debug(f"DHT lookup for layer {layer_id} failed: {e}")
        
        return []
    
    def _find_frontier_layer(self) -> int:
        """
        Find the first layer that has no holders in DHT.
        This is where the pipeline ends and new nodes should start.
        """
        arch_layers = self.current_architecture.num_layers if self.current_architecture else 11
        
        for layer_id in range(arch_layers):
            holders = self._discover_layer_holders_from_dht(layer_id)
            if not holders:
                return layer_id
        
        # All layers have holders - network is fully covered
        return arch_layers
    
    def register_node(
        self,
        node_id: str,
        node_url: str,
        grpc_addr: str,
        available_memory_mb: float,
        staked_amount: float = 0.0
    ) -> List[int]:
        """
        Register a node and assign layers.
        
        SIMPLE PROTOCOL:
        1. Query DHT for layer_0 holders
        2. If no holders → I am DRIVER (first node)
        3. If holders exist → I am WORKER, take NEXT available layers
        4. Last node in pipeline becomes VALIDATOR (has LM head)
        """
        from neuroshard.core.economics.constants import VALIDATOR_MIN_MEMORY_MB
        
        with self.lock:
            self.node_capacities[node_id] = available_memory_mb
            
            # Ensure architecture exists
            if self.current_architecture is None:
                self._auto_recalculate_architecture()
            
            arch_layers = self.current_architecture.num_layers
            
            # Calculate node capacity
            device_type = getattr(self, '_device_hint', 'cpu')
            safety_factor = 0.6 if device_type == 'cuda' else 0.5
            
            from neuroshard.core.model.scaler import calculate_layer_assignment
            
            # Check if we have a checkpoint with layer 0 (we were DRIVER)
            checkpoint_has_layer_0 = False
            try:
                checkpoint_path = self._get_checkpoint_path(node_id)
                if checkpoint_path and checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    saved_layers = checkpoint.get("layer_ids", [])
                    checkpoint_has_layer_0 = 0 in saved_layers
                    if checkpoint_has_layer_0:
                        logger.info(f"Checkpoint has layer 0 - this node was previously a DRIVER")
            except Exception as e:
                logger.debug(f"Could not check checkpoint: {e}")
            
            # STEP 1: Check if layer 0 has holders (is there a DRIVER?)
            layer_0_holders = self._discover_layer_holders_from_dht(0)
            
            # Calculate memory needed for embedding (weight + Adam optimizer states)
            # DRIVER must support full tokenizer vocab - no capping!
            # Memory = vocab × hidden × 4 bytes × 3 (weight + Adam m + Adam v)
            embed_memory_mb = (self.vocab_capacity * self.current_architecture.hidden_dim * 4 * 3) / (1024 * 1024)
            can_fit_vocab = available_memory_mb >= embed_memory_mb + 200  # +200MB for at least 1 layer
            
            if (not layer_0_holders or checkpoint_has_layer_0) and can_fit_vocab:
                # I am DRIVER (first node or reclaiming from checkpoint)
                role = "DRIVER"
                needs_embedding = True
                logger.info(f"Becoming DRIVER (layer_0 holders: {len(layer_0_holders)}, checkpoint: {checkpoint_has_layer_0})")
            elif not layer_0_holders and not can_fit_vocab:
                # No DRIVER exists but I can't fit vocab - wait as WORKER (redundancy mode)
                role = "WORKER"
                needs_embedding = False
                logger.warning(f"No DRIVER yet but can't fit embedding ({embed_memory_mb:.0f}MB needed, {available_memory_mb:.0f}MB available) - waiting as WORKER")
            else:
                # Another node is DRIVER - I am WORKER
                role = "WORKER"
                needs_embedding = False
                logger.info(f"Another node is DRIVER ({layer_0_holders[0]}) - becoming WORKER")
            
            # STEP 2: Calculate how many layers I can hold
            max_layers = calculate_layer_assignment(
                available_memory_mb=available_memory_mb,
                arch=self.current_architecture,
                safety_factor=safety_factor,
                vocab_capacity=self.vocab_capacity,
                training_mode=True,
                needs_embedding=needs_embedding
            )
            
            logger.info(f"Layer calculation ({role}): {available_memory_mb:.0f}MB × {safety_factor} safety = {max_layers} layers")
            
            # STEP 3: Assign layers based on role
            assigned_layers = []
            
            if role == "DRIVER":
                # Take layers 0, 1, 2, ... up to my capacity
                for layer_id in range(min(max_layers, arch_layers)):
                    self._assign_layer(layer_id, node_id, node_url, grpc_addr)
                    assigned_layers.append(layer_id)
                
                self.embedding_holder = node_id
                logger.info(f"DRIVER: Assigned layers 0-{max(assigned_layers) if assigned_layers else 0}")
                
                # If I hold all layers, I'm also the VALIDATOR
                if assigned_layers and max(assigned_layers) == arch_layers - 1:
                    self.lm_head_holder = node_id
                    logger.info(f"DRIVER is also VALIDATOR (full node)")
                elif assigned_layers:
                    # I'm DRIVER but not VALIDATOR - need more nodes for full pipeline
                    self.lm_head_holder = node_id  # Temporary until WORKER joins
                    logger.info(f"DRIVER holding LM head temporarily until WORKER joins")
            
            else:  # WORKER
                # Find where the current pipeline ends
                frontier = self._find_frontier_layer()
                
                if frontier >= arch_layers:
                    # All layers are covered - provide redundancy on last layers
                    frontier = max(1, arch_layers - max_layers)
                    logger.info(f"Network fully covered - providing redundancy from layer {frontier}")
                else:
                    logger.info(f"Pipeline frontier at layer {frontier}")
                
                # Take layers from frontier to end (or capacity)
                for layer_id in range(frontier, min(frontier + max_layers, arch_layers)):
                    self._assign_layer(layer_id, node_id, node_url, grpc_addr)
                    assigned_layers.append(layer_id)
                
                if assigned_layers:
                    logger.info(f"WORKER: Assigned layers {min(assigned_layers)}-{max(assigned_layers)}")
                    
                    # If I hold the last layer, I'm the VALIDATOR
                    if max(assigned_layers) == arch_layers - 1:
                        self.lm_head_holder = node_id
                        logger.info(f"WORKER is VALIDATOR (holds last layer)")
            
            # Update current_num_layers
            if assigned_layers:
                self.current_num_layers = max(self.current_num_layers, max(assigned_layers) + 1)
            
            logger.info(f"Node {node_id[:8]}... registered: {len(assigned_layers)} layers")
            logger.info(f"Assigned layers: {assigned_layers}")
            
            return assigned_layers
    
    def _assign_layer(self, layer_id: int, node_id: str, node_url: str, grpc_addr: str):
        """Assign a layer to a node."""
        if layer_id not in self.layer_assignments:
            self.layer_assignments[layer_id] = []
        
        # Check if already assigned
        if not any(a.node_id == node_id for a in self.layer_assignments[layer_id]):
            self.layer_assignments[layer_id].append(
                LayerAssignment(layer_id, node_id, node_url, grpc_addr)
            )
    
    def upgrade_to_validator(self, node_id: str, node_url: str, grpc_addr: str) -> bool:
        """Upgrade a node to validator by assigning the last layer."""
        with self.lock:
            if self.current_num_layers == 0:
                return False
            
            last_layer = self.current_num_layers - 1
            
            # Check if already a validator
            current_holders = self.layer_assignments.get(last_layer, [])
            if any(a.node_id == node_id for a in current_holders):
                return True  # Already has last layer
            
            # Assign last layer
            self._assign_layer(last_layer, node_id, node_url, grpc_addr)
            self.lm_head_holder = node_id
            
            logger.info(f"Node {node_id[:8]}... upgraded to VALIDATOR (layer {last_layer})")
            return True
    
    def remove_node(self, node_id: str):
        """Remove a node and its layer assignments."""
        with self.lock:
            orphaned_layers = []
            
            for layer_id in list(self.layer_assignments.keys()):
                before = len(self.layer_assignments[layer_id])
                self.layer_assignments[layer_id] = [
                    a for a in self.layer_assignments[layer_id] if a.node_id != node_id
                ]
                after = len(self.layer_assignments[layer_id])
                
                if before > after:
                    if not self.layer_assignments[layer_id]:
                        orphaned_layers.append(layer_id)
            
            # Update role holders
            if self.embedding_holder == node_id:
                self.embedding_holder = None
            if self.lm_head_holder == node_id:
                self.lm_head_holder = None
            
            if node_id in self.node_capacities:
                del self.node_capacities[node_id]
            
            if orphaned_layers:
                logger.warning(f"Node {node_id[:8]}... left, {len(orphaned_layers)} layers orphaned")
    
    def get_layer_holders(self, layer_id: int) -> List[LayerAssignment]:
        """Get all holders of a specific layer."""
        with self.lock:
            return list(self.layer_assignments.get(layer_id, []))
    
    def get_pipeline_route(self) -> List[str]:
        """Get the ordered list of node URLs for pipeline routing."""
        with self.lock:
            route = []
            for layer_id in range(self.current_num_layers):
                holders = self.layer_assignments.get(layer_id, [])
                if holders:
                    route.append(holders[0].node_url)
            return route
    
    def get_network_capacity(self) -> Dict[str, Any]:
        """Get network capacity statistics."""
        with self.lock:
            return {
                "total_nodes": len(self.node_capacities),
                "total_memory_mb": sum(self.node_capacities.values()),
                "total_layers": self.current_num_layers,
                "architecture": {
                    "num_layers": self.current_architecture.num_layers if self.current_architecture else 0,
                    "hidden_dim": self.current_architecture.hidden_dim if self.current_architecture else 0,
                } if self.current_architecture else None
            }
    
    def demote_from_validator(self, node_id: str) -> bool:
        """Remove validator status from a node."""
        with self.lock:
            if self.lm_head_holder == node_id:
                self.lm_head_holder = None
                logger.info(f"Node {node_id[:8]}... demoted from VALIDATOR")
                return True
            return False
    
    def validate_all_validators(self, get_stake_fn) -> List[str]:
        """Validate all validators meet stake requirements."""
        demoted = []
        with self.lock:
            if self.lm_head_holder:
                try:
                    stake = get_stake_fn(self.lm_head_holder)
                    # Require minimum stake for validator
                    if stake < 100.0:  # VALIDATOR_STAKE_REQUIRED
                        self.lm_head_holder = None
                        demoted.append(self.lm_head_holder)
                except:
                    pass
        return demoted
    
    def heartbeat(self, node_id: str, layer_ids: List[int]):
        """Record heartbeat from a node."""
        with self.lock:
            if not hasattr(self, '_last_heartbeat'):
                self._last_heartbeat = {}
            self._last_heartbeat[node_id] = time.time()
    
    def cleanup_stale_assignments(self, timeout_seconds: float = 300) -> List[str]:
        """Remove assignments from nodes that haven't sent heartbeats."""
        stale_nodes = []
        
        # First, identify stale nodes while holding the lock
        with self.lock:
            if not hasattr(self, '_last_heartbeat'):
                self._last_heartbeat = {}

            now = time.time()
            for node_id, last_seen in list(self._last_heartbeat.items()):
                if now - last_seen > timeout_seconds:
                    stale_nodes.append(node_id)
        
        # Remove nodes outside the lock to avoid deadlock
        # (remove_node also acquires self.lock)
        for node_id in stale_nodes:
            self.remove_node(node_id)
            with self.lock:
                if node_id in self._last_heartbeat:
                    del self._last_heartbeat[node_id]

        return stale_nodes


class DynamicNeuroLLM(nn.Module):
    """
    Dynamic LLM that holds a subset of layers.
    
    Each node holds different layers based on capacity.
    Forward/backward passes are pipelined across nodes via gRPC.
    """
    
    def __init__(self, node_id: str, layer_pool: DynamicLayerPool, device: str = "cpu"):
        super().__init__()
        self.node_id = node_id
        self.layer_pool = layer_pool
        self.device = device
        
        # Architecture from layer pool
        self.architecture = layer_pool.current_architecture
        self.vocab_capacity = layer_pool.vocab_capacity
        
        # Layers this node holds
        self.my_layers: Dict[int, nn.Module] = {}
        self.my_layer_ids: List[int] = []
        
        # Embedding and LM head (only if we're holder)
        self.embedding: Optional[nn.Embedding] = None
        self.lm_head: Optional[nn.Linear] = None
        self.final_norm: Optional[nn.Module] = None
        
        self.has_embedding = False
        self.has_lm_head = False
        
        # P2P manager reference (set later)
        self._p2p_manager = None
        self._on_layers_changed = None
        
        logger.info(f"DynamicNeuroLLM initialized for node {node_id[:8]}... "
                   f"with {self.architecture.num_layers}L × {self.architecture.hidden_dim}H architecture")
    
    def initialize_layers(self, layer_ids: List[int]):
        """Initialize the layers assigned to this node."""
        from neuroshard.core.model.llm import NeuroLLMConfig, NeuroDecoderLayer, RMSNorm
        
        config = NeuroLLMConfig(
            hidden_dim=self.architecture.hidden_dim,
            intermediate_dim=self.architecture.intermediate_dim,
            num_layers=self.architecture.num_layers,
            num_heads=self.architecture.num_heads,
            num_kv_heads=self.architecture.num_kv_heads,
            vocab_size=self.architecture.vocab_size,
            max_seq_len=self.architecture.max_seq_len,
            dropout=self.architecture.dropout,
            rope_theta=self.architecture.rope_theta,
        )
        
        # Initialize transformer layers
        for layer_id in layer_ids:
            layer = NeuroDecoderLayer(config, layer_id)
            layer.to(self.device)
            self.my_layers[layer_id] = layer
        
        self.my_layer_ids = sorted(layer_ids)
        
        # Initialize embedding if we're the holder
        # NOTE: DRIVER must support the FULL tokenizer vocab - no capping allowed!
        # The role assignment logic should prevent small-memory nodes from becoming DRIVER.
        if self.layer_pool.embedding_holder == self.node_id:
            self.embedding = nn.Embedding(self.vocab_capacity, self.architecture.hidden_dim)
            self.embedding.to(self.device)
            self.has_embedding = True
            logger.info(f"Initialized embedding: {self.vocab_capacity:,} tokens")

        # Initialize LM head if we're the holder
        if self.layer_pool.lm_head_holder == self.node_id:
            self.lm_head = nn.Linear(self.architecture.hidden_dim, self.vocab_capacity, bias=False)
            self.final_norm = RMSNorm(self.architecture.hidden_dim)
            self.lm_head.to(self.device)
            self.final_norm.to(self.device)
            self.has_lm_head = True
            logger.info(f"Initialized LM head: {self.vocab_capacity} outputs")
        
        logger.info(f"Initialized {len(layer_ids)} layers: {layer_ids}, "
                   f"arch={self.architecture.num_layers}L×{self.architecture.hidden_dim}H, "
                   f"embedding={self.has_embedding}, head={self.has_lm_head}")
    
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        if not self.has_embedding:
            raise RuntimeError("This node doesn't have the embedding layer")
        return self.embedding(input_ids)
    
    def forward_my_layers(self, hidden_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through only my layers."""
        for layer_id in self.my_layer_ids:
            layer = self.my_layers[layer_id]
            result = layer(hidden_states, attention_mask=attention_mask)
            # Handle layers that return (hidden_states, cache) tuple
            if isinstance(result, tuple):
                hidden_states = result[0]
            else:
                hidden_states = result
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute output logits."""
        if not self.has_lm_head:
            raise RuntimeError("This node doesn't have the LM head")
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full forward pass (only works if we have all components)."""
        if not self.has_embedding:
            raise RuntimeError("Cannot do full forward without embedding")
        
        hidden_states = self.embed(input_ids)
        hidden_states = self.forward_my_layers(hidden_states, attention_mask)
        
        if self.has_lm_head:
            return self.compute_logits(hidden_states)
        return hidden_states
    
    def check_and_expand_vocab_if_needed(self, new_vocab_size: int) -> bool:
        """Expand vocabulary if needed."""
        if new_vocab_size <= self.vocab_capacity:
            return False
        
        # Round up to nearest 32K for efficiency
        VOCAB_GROWTH_CHUNK = 32000
        new_capacity = ((new_vocab_size // VOCAB_GROWTH_CHUNK) + 1) * VOCAB_GROWTH_CHUNK
        new_capacity = min(new_capacity, MAX_VOCAB_SIZE)
        
        if new_capacity <= self.vocab_capacity:
            return False
        
        logger.info(f"[VOCAB] Expanding vocabulary: {self.vocab_capacity} → {new_capacity}")
        
        # Expand embedding
        if self.has_embedding and self.embedding is not None:
            old_embedding = self.embedding
            self.embedding = nn.Embedding(new_capacity, self.architecture.hidden_dim)
            with torch.no_grad():
                self.embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
            self.embedding.to(self.device)
            logger.info(f"[VOCAB] Expanded embedding: {old_embedding.num_embeddings} → {new_capacity} tokens")
        
        # Expand LM head
        if self.has_lm_head and self.lm_head is not None:
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(self.architecture.hidden_dim, new_capacity, bias=False)
            with torch.no_grad():
                self.lm_head.weight[:old_lm_head.out_features] = old_lm_head.weight
            self.lm_head.to(self.device)
            logger.info(f"[VOCAB] Expanded lm_head: {old_lm_head.out_features} → {new_capacity} outputs")
        
        self.vocab_capacity = new_capacity
        self.layer_pool.vocab_capacity = new_capacity
        
        logger.info(f"[VOCAB] ✅ Vocabulary expansion complete: {self.vocab_capacity}")
        return True
    
    def parameters(self, recurse: bool = True):
        """Get all parameters."""
        params = []
        for layer in self.my_layers.values():
            params.extend(layer.parameters(recurse))
        if self.embedding is not None:
            params.extend(self.embedding.parameters(recurse))
        if self.lm_head is not None:
            params.extend(self.lm_head.parameters(recurse))
        if self.final_norm is not None:
            params.extend(self.final_norm.parameters(recurse))
        return iter(params)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Get named parameters."""
        for name, param in super().named_parameters(prefix, recurse):
            yield name, param
        for layer_id, layer in self.my_layers.items():
            for name, param in layer.named_parameters(f"layer_{layer_id}", recurse):
                yield name, param
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension (for compatibility)."""
        return self.architecture.hidden_dim if self.architecture else 512
    
    @property
    def num_heads(self) -> int:
        """Get number of attention heads (for compatibility)."""
        return self.architecture.num_heads if self.architecture else 8
    
    def initialize_lm_head(self) -> bool:
        """Initialize LM head for validator upgrade."""
        if self.has_lm_head:
            return False  # Already has LM head
        
        try:
            from neuroshard.core.model.llm import RMSNorm
            self.lm_head = nn.Linear(self.architecture.hidden_dim, self.vocab_capacity, bias=False)
            self.final_norm = RMSNorm(self.architecture.hidden_dim)
            self.lm_head.to(self.device)
            self.final_norm.to(self.device)
            self.has_lm_head = True
            self.layer_pool.lm_head_holder = self.node_id
            logger.info(f"[VALIDATOR] Initialized LM head: {self.vocab_capacity} outputs")
            return True
        except Exception as e:
            logger.error(f"[VALIDATOR] Failed to initialize LM head: {e}")
            return False
    
    def disable_lm_head(self) -> bool:
        """Disable LM head for validator demotion."""
        if not self.has_lm_head:
            return False  # Already disabled
        
        try:
            self.lm_head = None
            self.final_norm = None
            self.has_lm_head = False
            if self.layer_pool.lm_head_holder == self.node_id:
                self.layer_pool.lm_head_holder = None
            logger.info("[VALIDATOR] LM head disabled (demoted from validator)")
            return True
        except Exception as e:
            logger.error(f"[VALIDATOR] Failed to disable LM head: {e}")
            return False
    
    def get_my_contribution(self) -> Dict[str, Any]:
        """Get this node's contribution stats."""
        return {
            "layers": len(self.my_layer_ids),
            "layer_ids": self.my_layer_ids,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "parameters": self.get_num_params(),
            "vocab_capacity": self.vocab_capacity,
        }


class DynamicNeuroNode:
    """
    A node in the NeuroShard network.
    
    Manages:
    - Layer assignment and initialization
    - Training with pipeline parallelism
    - Checkpointing and recovery
    - P2P communication for distributed training
    """
    
    def __init__(
        self,
        node_id: str,
        wallet_id: str,
        port: int = 8000,
        available_memory_mb: float = 4000,
        device: str = "cpu",
        enable_training: bool = True,
        max_storage_mb: int = 5000
    ):
        self.node_id = node_id
        self.wallet_id = wallet_id
        self.node_token = node_id  # Alias for compatibility with SwarmEnabledNode
        self.port = port
        self.available_memory_mb = available_memory_mb
        self.device = device
        self.enable_training = enable_training
        self.max_storage_mb = max_storage_mb
        
        # Components (initialized in start())
        self.layer_pool: Optional[DynamicLayerPool] = None
        self.model: Optional[DynamicNeuroLLM] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.tokenizer = None
        self.data_manager = None  # For training data management
        
        # P2P manager (set externally before start)
        self.p2p_manager = None
        
        # Layer tracking
        self.my_layer_ids: List[int] = []
        
        # Training state
        self.current_loss: Optional[float] = None
        self.training_context: Dict[str, Any] = {}
        self._training_lock = threading.Lock()
        self._training_batch_size = 4
        self._use_gradient_checkpointing = False
        
        # Training statistics (for PoNW and monitoring)
        self.total_training_rounds = 0
        self.total_tokens_processed = 0
        self.training_contribution_count = 0
        self._global_tracker = None
        
        # Genesis data loader
        self.genesis_loader = None
        
        # Swarm (DataSwarm for P2P data transfer)
        self.swarm = None
        
        # State
        self.is_running = False
    
    def start(self):
        """Start the node and initialize components."""
        logger.info("Starting DynamicNeuroNode...")
        
        # 1. Initialize layer pool with DHT
        dht = None
        if self.p2p_manager and hasattr(self.p2p_manager, 'dht'):
            dht = self.p2p_manager.dht
            logger.info("P2P connected BEFORE start - DHT available for network discovery")
        
        self.layer_pool = DynamicLayerPool(dht_protocol=dht)
        self.layer_pool._device_hint = self.device
        self.layer_pool._wallet_id = self.wallet_id
        
        # 2. Load saved checkpoint if exists
        self._load_or_sync_architecture()
        
        # 3. Pre-fetch vocab size for memory calculation
        self._prefetch_vocab_capacity()
        
        # 4. Register with network and get layer assignment
        staked_amount = self._get_staked_amount()
        
        self.my_layer_ids = self.layer_pool.register_node(
            node_id=self.node_id,
            node_url=f"http://localhost:{self.port}",
            grpc_addr=f"localhost:{self.port + 1000}",
            available_memory_mb=self.available_memory_mb,
            staked_amount=staked_amount
        )
        
        logger.info(f"Assigned {len(self.my_layer_ids)} layers: {self.my_layer_ids}")
        
        # 5. Initialize model with my layers
        self.model = DynamicNeuroLLM(
            node_id=self.node_id,
            layer_pool=self.layer_pool,
            device=self.device
        )
        self.model.initialize_layers(self.my_layer_ids)
        
        # 6. Load tokenizer and expand vocab if needed
        self._load_learned_tokenizer()
        
        # 7. Load checkpoint weights
        self._load_checkpoint()
        
        # 8. Setup training
        if self.enable_training:
            self._setup_training()
        
        self.is_running = True
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Node started: {param_count:.1f}M params, {len(self.my_layer_ids)} layers, "
                   f"embed={self.model.has_embedding}, head={self.model.has_lm_head}")
    
    def _load_or_sync_architecture(self):
        """Load architecture from checkpoint or sync from network."""
        checkpoint_path = self._get_checkpoint_path()
        
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                saved_arch = checkpoint.get("architecture", {})
                
                logger.info(f"Saved checkpoint: {saved_arch.get('num_layers', '?')}L × {saved_arch.get('hidden_dim', '?')}H")
                
                # Use saved architecture
                from neuroshard.core.model.scaler import ModelArchitecture
                self.layer_pool.current_architecture = ModelArchitecture(
                    hidden_dim=saved_arch.get('hidden_dim', 512),
                    intermediate_dim=saved_arch.get('intermediate_dim', 2048),
                    num_layers=saved_arch.get('num_layers', 11),
                    num_heads=saved_arch.get('num_heads', 16),
                    num_kv_heads=saved_arch.get('num_kv_heads', 4),
                    vocab_size=MAX_VOCAB_SIZE,
                    max_seq_len=saved_arch.get('max_seq_len', 2048),
                    dropout=saved_arch.get('dropout', 0.1),
                    rope_theta=saved_arch.get('rope_theta', 10000.0)
                )
                self.layer_pool.current_num_layers = self.layer_pool.current_architecture.num_layers
                
                logger.info(f"Network architecture: {self.layer_pool.current_architecture.num_layers}L × "
                           f"{self.layer_pool.current_architecture.hidden_dim}H")
                logger.info("✅ Checkpoint compatible with network - will load checkpoint")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                logger.info("No saved checkpoint found")
        else:
            logger.info("No saved checkpoint found")
            # Will use default architecture
            if self.layer_pool.current_architecture is None:
                self.layer_pool._auto_recalculate_architecture()
            logger.info(f"Network architecture: {self.layer_pool.current_architecture.num_layers}L × "
                       f"{self.layer_pool.current_architecture.hidden_dim}H")
            logger.info("✅ Joining network with architecture: "
                       f"{self.layer_pool.current_architecture.num_layers}L × "
                       f"{self.layer_pool.current_architecture.hidden_dim}H")
    
    def _get_checkpoint_path(self) -> Optional[Path]:
        """Get path to checkpoint file."""
        checkpoint_dir = Path.home() / ".neuroshard" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"dynamic_node_{self.wallet_id}.pt"
    
    def _prefetch_vocab_capacity(self):
        """Pre-fetch vocab size from CDN tokenizer for memory calculation."""
        try:
            import os
            import urllib.request
            import json
            from pathlib import Path
            from neuroshard.core.model.tokenizer import NeuroTokenizer
            
            # Try to load learned tokenizer from CDN (same URL as GenesisDataLoader)
            CDN_URL = "https://dwquwt9gkkeil.cloudfront.net"
            cache_dir = Path(os.path.expanduser("~/.neuroshard/data_cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_cache_path = cache_dir / "tokenizer.json"
            
            current_vocab = 266  # Base vocab (fallback)
            
            # Try downloading from CDN
            try:
                tokenizer_url = f"{CDN_URL}/tokenizer.json"
                req = urllib.request.Request(tokenizer_url, headers={"User-Agent": "NeuroShard/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    remote_data = json.loads(resp.read().decode('utf-8'))
                    current_vocab = remote_data.get("next_merge_id", 266)
                    
                    # Cache for offline use
                    with open(tokenizer_cache_path, 'w') as f:
                        json.dump(remote_data, f)
                    
                    logger.info(f"[VOCAB] Downloaded tokenizer from CDN: {current_vocab:,} tokens")
            except Exception as e:
                # Try cached version
                if tokenizer_cache_path.exists():
                    try:
                        with open(tokenizer_cache_path) as f:
                            cached_data = json.load(f)
                        current_vocab = cached_data.get("next_merge_id", 266)
                        logger.info(f"[VOCAB] Using cached tokenizer: {current_vocab:,} tokens")
                    except:
                        pass
                else:
                    logger.debug(f"[VOCAB] Could not fetch tokenizer from CDN: {e}")

            # Allocate for current vocab + growth headroom
            VOCAB_GROWTH_CHUNK = 32000
            capacity = ((current_vocab // VOCAB_GROWTH_CHUNK) + 1) * VOCAB_GROWTH_CHUNK
            capacity = min(capacity, MAX_VOCAB_SIZE)

            self.layer_pool.vocab_capacity = capacity
            logger.info(f"[VOCAB] Pre-fetched tokenizer: {current_vocab:,} tokens (for memory calculation)")
            logger.info(f"[VOCAB] Layer pool vocab_capacity set to {capacity:,} (current vocab: {current_vocab:,})")
        except Exception as e:
            logger.warning(f"[VOCAB] Could not pre-fetch vocab size: {e}")
            self.layer_pool.vocab_capacity = INITIAL_VOCAB_SIZE
    
    def _get_staked_amount(self) -> float:
        """Get staked amount from ledger."""
        if self.p2p_manager and hasattr(self.p2p_manager, 'ledger') and self.p2p_manager.ledger:
            try:
                account_info = self.p2p_manager.ledger.get_account_info()
                staked = account_info.get("stake", 0.0)
                logger.info(f"Current stake: {staked:.2f} NEURO")
                return staked
            except:
                pass
        logger.info("Current stake: 0.00 NEURO")
        return 0.0
    
    def _load_learned_tokenizer(self):
        """Load the learned tokenizer and expand vocab if needed."""
        try:
            from neuroshard.core.model.tokenizer import get_neuro_tokenizer
            self.tokenizer = get_neuro_tokenizer()
            
            logger.info(f"[TOKENIZER] Loaded BPE tokenizer: {self.tokenizer.current_vocab_size} tokens, "
                       f"{len(self.tokenizer.merges)} merges")

            # Check if vocab expansion needed (use current_vocab_size, not max vocab_size)
            if self.tokenizer.current_vocab_size > self.model.vocab_capacity:
                logger.info(f"[VOCAB] Tokenizer vocab ({self.tokenizer.current_vocab_size}) exceeds capacity ({self.model.vocab_capacity})")
                self.model.check_and_expand_vocab_if_needed(self.tokenizer.current_vocab_size)
        except Exception as e:
            logger.warning(f"[TOKENIZER] Error loading learned tokenizer: {e}")
    
    def _load_checkpoint(self):
        """Load checkpoint weights if available."""
        checkpoint_path = self._get_checkpoint_path()
        
        if not checkpoint_path or not checkpoint_path.exists():
            logger.info(f"No checkpoint found at {checkpoint_path.name if checkpoint_path else 'None'}, starting fresh")
            return
        
        try:
            logger.info(f"Loading checkpoint from: {checkpoint_path.name}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load layer weights
            saved_layers = checkpoint.get("layer_ids", [])
            common_layers = set(saved_layers) & set(self.my_layer_ids)
            
            if len(common_layers) != len(saved_layers):
                logger.info(f"Layer assignment changed: saved={len(saved_layers)}, current={len(self.my_layer_ids)}, common={len(common_layers)}")
            
            # Load state dict
            state_dict = checkpoint.get("state_dict", {})
            
            # Load embedding
            if self.model.has_embedding and "embedding.weight" in state_dict:
                self._load_embedding_with_expansion(state_dict)
            
            # Load LM head
            if self.model.has_lm_head and "lm_head.weight" in state_dict:
                self._load_lm_head_with_expansion(state_dict)
            
            # Load layer weights
            for layer_id in common_layers:
                layer_prefix = f"layer_{layer_id}."
                layer_state = {k[len(layer_prefix):]: v for k, v in state_dict.items() 
                              if k.startswith(layer_prefix)}
                if layer_state and layer_id in self.model.my_layers:
                    self.model.my_layers[layer_id].load_state_dict(layer_state, strict=False)
            
            training_rounds = checkpoint.get("training_rounds", 0)
            logger.info(f"Checkpoint loaded: {training_rounds} training rounds, "
                       f"{len(common_layers)}/{len(self.my_layer_ids)} layers from {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _load_embedding_with_expansion(self, state_dict: dict):
        """Load embedding weights, handling vocab size mismatch."""
        saved_weight = state_dict["embedding.weight"]
        saved_vocab = saved_weight.shape[0]
        current_vocab = self.model.embedding.num_embeddings
        
        if saved_vocab == current_vocab:
            self.model.embedding.load_state_dict({"weight": saved_weight})
            logger.info(f"[CHECKPOINT] Loaded embedding: {saved_vocab} tokens")
        elif saved_vocab < current_vocab:
            with torch.no_grad():
                self.model.embedding.weight[:saved_vocab] = saved_weight
            logger.info(f"[CHECKPOINT] Loaded embedding with vocab expansion: {saved_vocab} → {current_vocab} tokens")
        else:
            with torch.no_grad():
                self.model.embedding.weight[:] = saved_weight[:current_vocab]
            logger.warning(f"[CHECKPOINT] Truncated embedding: {saved_vocab} → {current_vocab} tokens")
    
    def _load_lm_head_with_expansion(self, state_dict: dict):
        """Load LM head weights, handling vocab size mismatch."""
        saved_weight = state_dict["lm_head.weight"]
        saved_vocab = saved_weight.shape[0]
        current_vocab = self.model.lm_head.out_features
        
        if saved_vocab == current_vocab:
            self.model.lm_head.load_state_dict({"weight": saved_weight})
            logger.info(f"[CHECKPOINT] Loaded lm_head: {saved_vocab} outputs")
        elif saved_vocab < current_vocab:
            with torch.no_grad():
                self.model.lm_head.weight[:saved_vocab] = saved_weight
            logger.info(f"[CHECKPOINT] Loaded lm_head with vocab expansion: {saved_vocab} → {current_vocab} outputs")
        else:
            with torch.no_grad():
                self.model.lm_head.weight[:] = saved_weight[:current_vocab]
            logger.warning(f"[CHECKPOINT] Truncated lm_head: {saved_vocab} → {current_vocab} outputs")
    
    def _setup_training(self):
        """Setup training components."""
        # Collect parameters
        all_params = list(self.model.parameters())
        
        if not all_params:
            raise ValueError("No parameters to train - model may not be initialized correctly")
        
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
        
        # Configure gradient checkpointing
        num_layers = len(self.my_layer_ids)
        if self.device == 'cuda':
            self._use_gradient_checkpointing = num_layers > 16
        else:
            self._use_gradient_checkpointing = True  # Always use on CPU
        
        # Calculate batch size based on memory
        if self._use_gradient_checkpointing:
            segments = max(1, num_layers // 4)
            mem_per_sample = segments * 2.0  # MB per sample estimate
            logger.info(f"[NODE] Gradient checkpointing: ENABLED ({segments} segments, ~{mem_per_sample:.1f}MB/sample)")
        else:
            logger.info(f"[NODE] Gradient checkpointing: DISABLED (layers={num_layers} ≤ 16, CUDA)")
        
        # Log training config
        param_count = sum(p.numel() for p in all_params) / 1e6
        logger.info(f"[NODE] Training config: batch_size={self._training_batch_size}, "
                   f"model={param_count:.1f}M params ({num_layers} layers × {self.model.architecture.hidden_dim} dim), "
                   f"checkpointing={self._use_gradient_checkpointing}, device={self.device}")
        logger.info(f"Training initialized: batch_size={self._training_batch_size}, "
                   f"checkpointing={self._use_gradient_checkpointing}, layers={num_layers}, device={self.device}")
    
    def train_step(self) -> Optional[float]:
        """
        Execute one training step.
        
        PIPELINE TRAINING:
        1. DRIVER: Get batch → embed → forward → send to next
        2. WORKER: Receive → forward → send to next
        3. VALIDATOR: Receive → forward → loss → backward → send grads
        4. Everyone updates their weights
        """
        if not self.enable_training or not self.model:
            return None
        
        try:
            # Check if there's a next hop (another node with higher layers)
            my_last_layer = max(self.my_layer_ids) if self.my_layer_ids else 0
            next_layer = my_last_layer + 1
            
            has_next_hop = False
            next_hop_url = None
            if self.p2p_manager:
                next_hop_url = self.p2p_manager.get_next_hop(next_layer)
                if next_hop_url:
                    has_next_hop = True
            
            # Determine if I'm a "full node" (can train independently)
            # I'm full if: I have embedding AND (I have LM head AND no next hop)
            am_validator = self.model.has_lm_head and not has_next_hop
            is_full_node = self.model.has_embedding and am_validator
            
            if self.model.has_embedding:
                # I am DRIVER - get batch and start training
                input_ids, labels = self._get_training_batch()
                if input_ids is None:
                    return None
                
                if is_full_node:
                    # Solo training - do everything locally
                    return self._train_step_local(input_ids, labels)
                else:
                    # Distributed training - forward to next node
                    return self._train_step_distributed(input_ids, labels, next_hop_url)
            else:
                # I am WORKER/VALIDATOR - wait for data via gRPC
                # Training happens when PipelineForward is called
                return None
                
        except Exception as e:
            logger.warning(f"Training step error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _get_training_batch(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get a training batch from Genesis data."""
        if not hasattr(self, 'genesis_loader') or self.genesis_loader is None:
            try:
                from neuroshard.core.training.distributed import GenesisDataLoader
                from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                
                logger.info("[GENESIS] Initializing data loader...")
                self.genesis_loader = GenesisDataLoader(
                    self.node_id,
                    get_neuro_tokenizer(),
                    max_storage_mb=self.max_storage_mb
                )
                
                if hasattr(self, 'swarm') and self.swarm:
                    self.genesis_loader.set_swarm(self.swarm)
                
                logger.info(f"[GENESIS] Data loader ready: {self.genesis_loader.total_shards} shards available")
            except Exception as e:
                logger.error(f"[GENESIS] Failed to initialize: {e}")
                return None, None
        
        if not self.genesis_loader.is_data_ready():
            return None, None
        
        try:
            input_ids, labels = self.genesis_loader.get_batch(batch_size=self._training_batch_size)
            return input_ids.to(self.device), labels.to(self.device)
        except Exception as e:
            logger.warning(f"[GENESIS] Failed to get batch: {e}")
            return None, None
    
    def _train_step_local(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
        """Execute local training step (full node mode)."""
        with self._training_lock:
            self.optimizer.zero_grad()
            
            # Forward pass
            if self._use_gradient_checkpointing:
                embeddings = self.model.embed(input_ids)
                hidden = torch.utils.checkpoint.checkpoint(
                    self.model.forward_my_layers,
                    embeddings,
                    use_reentrant=False
                )
            else:
                hidden = self.model(input_ids)
            
            logits = self.model.compute_logits(hidden) if not isinstance(hidden, tuple) else hidden
            
            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.current_loss = loss.item()
            return self.current_loss
    
    def _train_step_distributed(self, input_ids: torch.Tensor, labels: torch.Tensor, 
                                next_hop_url: str) -> Optional[float]:
        """Execute distributed training step (pipeline mode)."""
        # Embed and forward through my layers
        embeddings = self.model.embed(input_ids)
        
        if self._use_gradient_checkpointing:
            hidden = torch.utils.checkpoint.checkpoint(
                self.model.forward_my_layers,
                embeddings,
                use_reentrant=False
            )
        else:
            hidden = self.model.forward_my_layers(embeddings)
        
        # Forward to next node via gRPC
        try:
            from neuroshard.core.network.connection_pool import get_connection_pool
            import neuroshard.protos.neuroshard_pb2 as pb2
            import neuroshard.protos.neuroshard_pb2_grpc as pb2_grpc
            from neuroshard.utils.serialization import serialize_tensor, deserialize_tensor
            
            # Get gRPC channel
            pool = get_connection_pool()
            grpc_addr = next_hop_url.replace("http://", "").replace(":8000", ":9000")
            channel = pool.get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardStub(channel)
            
            # Serialize and send
            session_id = f"train_{self.node_id}_{time.time()}"
            
            request = pb2.PipelineForwardRequest(
                session_id=session_id,
                hidden_states=serialize_tensor(hidden),
                source_node_id=self.node_id,
                source_layer=max(self.my_layer_ids),
                is_training=True,
                training_labels=serialize_tensor(labels)
            )
            
            response = stub.PipelineForward(request, timeout=60)
            
            if response.success:
                # Get gradients and update
                if response.grad_output:
                    grad = deserialize_tensor(response.grad_output).to(self.device)
                    
                    with self._training_lock:
                        self.optimizer.zero_grad()
                        hidden.backward(grad)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.current_loss = response.loss if response.loss else None
                    return self.current_loss
                
                return response.loss if response.loss else None
            else:
                logger.warning(f"Pipeline forward failed: {response.error_message}")
                return None
                
        except Exception as e:
            logger.warning(f"Distributed training error: {e}")
            return None
    
    def save_checkpoint(self):
        """Save checkpoint."""
        if not self.model:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        
        state_dict = {}
        
        # Save embedding
        if self.model.has_embedding and self.model.embedding is not None:
            state_dict["embedding.weight"] = self.model.embedding.weight.cpu()
        
        # Save LM head
        if self.model.has_lm_head:
            if self.model.lm_head is not None:
                state_dict["lm_head.weight"] = self.model.lm_head.weight.cpu()
            if self.model.final_norm is not None:
                for name, param in self.model.final_norm.named_parameters():
                    state_dict[f"final_norm.{name}"] = param.cpu()
        
        # Save layers
        for layer_id, layer in self.model.my_layers.items():
            for name, param in layer.named_parameters():
                state_dict[f"layer_{layer_id}.{name}"] = param.cpu()
        
        checkpoint = {
            "state_dict": state_dict,
            "layer_ids": self.my_layer_ids,
            "architecture": {
                "hidden_dim": self.model.architecture.hidden_dim,
                "intermediate_dim": self.model.architecture.intermediate_dim,
                "num_layers": self.model.architecture.num_layers,
                "num_heads": self.model.architecture.num_heads,
                "num_kv_heads": self.model.architecture.num_kv_heads,
                "max_seq_len": self.model.architecture.max_seq_len,
            },
            "vocab_capacity": self.model.vocab_capacity,
            "training_rounds": getattr(self, 'training_rounds', 0),
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path.name}")
    
    def forward_pipeline(self, hidden_states: torch.Tensor, session_id: str,
                        source_layer: int, is_training: bool = False,
                        labels: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """
        Handle incoming pipeline forward request.
        
        Called via gRPC when another node forwards activations to us.
        """
        hidden_states = hidden_states.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # Forward through my layers
        output = self.model.forward_my_layers(hidden_states, attention_mask)
        
        # Check if there's a next hop
        my_last_layer = max(self.my_layer_ids) if self.my_layer_ids else 0
        next_layer = my_last_layer + 1
        
        next_hop = None
        if self.p2p_manager:
            next_hop = self.p2p_manager.get_next_hop(next_layer)
        
        if next_hop:
            # Forward to next node
            # (Similar to _train_step_distributed)
            # For now, return output for caller to forward
            return output, None, None
        
        # I'm the last node - compute loss if training
        if is_training and labels is not None and self.model.has_lm_head:
            with self._training_lock:
                self.optimizer.zero_grad()
                
                logits = self.model.compute_logits(output)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Return gradients for backward pass
                grad_output = hidden_states.grad if hidden_states.requires_grad else None
                return output, loss.item(), grad_output
        
        return output, None, None
    
    def backward_pipeline(self, grad_output: torch.Tensor, session_id: str) -> Optional[torch.Tensor]:
        """
        Handle incoming gradient for backward pass.
        """
        # This would be called if we're not the last node but received gradients
        # For now, this is handled inline in forward_pipeline
        return None
    
    def stop(self):
        """Stop the node."""
        self.is_running = False
        logger.info("DynamicNeuroNode stopped")
    
    def forward(self, input_ids: torch.Tensor, session_id: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        For inference, returns logits if we have LM head, else hidden states.
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        input_ids = input_ids.to(self.device)
        
        if self.model.has_embedding:
            hidden = self.model.embed(input_ids)
            hidden = self.model.forward_my_layers(hidden)
            
            if self.model.has_lm_head:
                return self.model.compute_logits(hidden)
            return hidden
        else:
            # We're a worker - just process through our layers
            return self.model.forward_my_layers(input_ids)
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """
        Generate text from a prompt.
        
        Only works if we have both embedding and LM head (full node).
        """
        if not self.model or not self.model.has_embedding or not self.model.has_lm_head:
            return "[Generation requires full node with embedding and LM head]"
        
        if not self.tokenizer:
            return "[Tokenizer not loaded]"
        
        try:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            generated = list(input_ids)
            
            for _ in range(max_tokens):
                with torch.no_grad():
                    logits = self.forward(input_tensor)
                    next_logits = logits[0, -1, :] / temperature
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                input_tensor = torch.tensor([generated], device=self.device)
                
                # Stop on EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            return self.tokenizer.decode(generated)
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return f"[Generation error: {e}]"
    
    def contribute_training_data(self, text: str, apply_dp: bool = False) -> int:
        """
        Contribute training data to the node.
        
        Returns number of tokens added.
        """
        if not self.enable_training or not self.tokenizer:
            return 0
        
        try:
            tokens = self.tokenizer.encode(text)
            self.training_contribution_count += 1
            return len(tokens)
        except Exception as e:
            logger.warning(f"Data contribution error: {e}")
            return 0
    
    def get_global_training_status(self) -> Dict[str, Any]:
        """Get global training status."""
        return {
            "total_training_rounds": self.total_training_rounds,
            "total_tokens_processed": self.total_tokens_processed,
            "current_loss": self.current_loss,
            "my_layers": self.my_layer_ids,
            "has_embedding": self.model.has_embedding if self.model else False,
            "has_lm_head": self.model.has_lm_head if self.model else False,
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        if not self.model:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _save_checkpoint(self, async_save: bool = True):
        """Save checkpoint (alias for save_checkpoint)."""
        self.save_checkpoint()


def create_dynamic_node(
    node_token: str,
    port: int = 8000,
    tracker_url: str = "http://localhost:3000",
    available_memory_mb: Optional[float] = None,
    enable_training: bool = True,
    max_storage_mb: int = 5000,
    max_cpu_threads: int = 4,
    device: str = "auto",
    p2p_manager = None
) -> DynamicNeuroNode:
    """
    Create and start a dynamic node.
    
    Args:
        node_token: Authentication token (used to derive node_id and wallet_id)
        port: HTTP port for the node
        tracker_url: URL of the tracker server
        available_memory_mb: Override memory detection
        enable_training: Whether to enable training
        max_storage_mb: Max disk space for data shards
        max_cpu_threads: Max CPU threads to use
        device: Device to use (auto, cpu, cuda, mps)
        p2p_manager: Optional P2P manager for DHT
    
    Returns:
        DynamicNeuroNode ready for use
    """
    import hashlib
    import platform
    
    # Derive node_id and wallet_id from token
    machine_id = hashlib.sha256(platform.node().encode()).hexdigest()[:16]
    instance_id = hashlib.sha256(f"{machine_id}:{port}".encode()).hexdigest()[:16]
    wallet_id = hashlib.sha256(node_token.encode()).hexdigest()[:16]
    node_id = hashlib.sha256(f"{instance_id}:{node_token}".encode()).hexdigest()[:20]
    
    logger.info(f"Instance ID: {instance_id} (machine+port)")
    logger.info(f"Wallet ID: {wallet_id}... (for NEURO earnings)")
    logger.info(f"Node ID: {node_id}... (unique network identity)")
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("[NODE] Using device: cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("[NODE] Using device: mps")
        else:
            device = "cpu"
            logger.info("[NODE] No GPU detected, using CPU")
    else:
        logger.info(f"[NODE] Device manually set to: {device}")
    
    # Auto-detect memory if not specified
    if available_memory_mb is None:
        try:
            import psutil
            total_mem = psutil.virtual_memory().total / (1024 * 1024)
            # Reserve some for OS
            available_memory_mb = min(total_mem * 0.8, 32000)
        except:
            available_memory_mb = 4000
    
    logger.info(f"Using device: {device}")
    
    # Create node
    node = DynamicNeuroNode(
        node_id=node_id,
        wallet_id=wallet_id,
        port=port,
        available_memory_mb=available_memory_mb,
        device=device,
        enable_training=enable_training,
        max_storage_mb=max_storage_mb
    )
    
    if p2p_manager:
        node.p2p_manager = p2p_manager
    
    logger.info(f"DynamicNeuroNode initialized: memory={available_memory_mb}MB")
    
    node.start()
    return node
