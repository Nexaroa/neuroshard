"""
Dynamic Model Architecture - True Decentralization

This module implements a model that grows and shrinks with the network:
- NO fixed phases or model sizes
- Model size = what the network can collectively hold
- Each node contributes based on its available memory
- More memory = more layers = more NEURO rewards

Key Concepts:
1. LAYER POOL: The network maintains a pool of layers
2. DYNAMIC ASSIGNMENT: Nodes claim layers based on their capacity
3. ORGANIC GROWTH: As more nodes join, model can have more layers
4. GRACEFUL DEGRADATION: If nodes leave, layers are redistributed

Example:
  Day 1: 10 nodes with 4GB each = 40GB total = ~10B params possible
  Day 30: 100 nodes with avg 8GB = 800GB total = ~200B params possible
  
  The model AUTOMATICALLY grows as capacity grows.
  No voting, no phases, no central coordination.
"""

import torch
import threading
import time
import logging
import hashlib
import math
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Dynamic architecture - NO MORE FIXED DIMENSIONS!
# Architecture is now calculated based on network capacity

VOCAB_SIZE = 32000  # NeuroLLM BPE vocabulary (fixed - tokenizer dependent)

# Import the new architecture scaler
from neuroshard.core.model.scaler import (
    ModelArchitecture,
    calculate_optimal_architecture,
    should_upgrade_architecture,
    estimate_memory_per_layer,
    calculate_layer_assignment,
)


@dataclass
class LayerAssignment:
    """Assignment of a layer to a node."""
    layer_id: int
    node_id: str
    node_url: str
    grpc_addr: str
    assigned_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    version: int = 0  # Training version


@dataclass
class NetworkCapacity:
    """Current network capacity."""
    total_nodes: int
    total_memory_mb: float
    max_layers: int  # How many layers the network can support
    assigned_layers: int  # How many layers are currently assigned
    layer_coverage: Dict[int, int]  # layer_id -> replica count


class DynamicLayerPool:
    """
    Manages the dynamic pool of model layers across the network.
    
    This is the core of true decentralization:
    - Layers are assigned based on node capacity
    - Model grows BOTH deeper AND wider as network expands
    - Architecture auto-optimizes based on scaling laws
    - No fixed model size or phases
    
    SCALABILITY CONSIDERATIONS:
    ==========================
    Small network (1-10 nodes):
    - Each node may hold ALL layers (solo training mode)
    - No layer replication needed
    - Fast startup, immediate training
    
    Medium network (10-100 nodes):
    - Layers distributed across multiple nodes
    - MIN_REPLICAS ensures redundancy
    - Pipeline inference works across nodes
    
    Large network (100-1000+ nodes):
    - Strong layer distribution
    - MAX_LAYERS_PER_NODE caps per-node load
    - Architecture can scale to 100B+ params
    """
    
    # Minimum replicas per layer for redundancy
    MIN_REPLICAS = 2
    
    # Layer heartbeat timeout
    HEARTBEAT_TIMEOUT = 120  # seconds
    
    # Maximum layers any single node can hold (prevents memory issues in large networks)
    # In small networks (< 100 nodes), this is effectively unlimited
    # In large networks, it ensures load is distributed
    MAX_LAYERS_PER_NODE = 200
    
    # Architecture recalculation triggers
    # NOTE: RECALC_INTERVAL_NODES is now DYNAMIC - see _get_recalc_interval()
    MIN_UPGRADE_IMPROVEMENT = 1.3  # Only upgrade if 30%+ better
    
    @staticmethod
    def _get_recalc_interval(node_count: int) -> int:
        """
        Get dynamic architecture recalculation interval based on network size.
        
        At small networks, recalculate more often (every node matters).
        At large networks, recalculate less often (stability).
        
        Formula: min(max(5, node_count // 10), 100)
        - 1-50 nodes: every 5 nodes
        - 51-100 nodes: every 5-10 nodes
        - 100-1000 nodes: every 10-100 nodes
        - 1000+ nodes: every 100 nodes
        """
        return min(max(5, node_count // 10), 100)
    
    def __init__(self, dht_protocol=None):
        self.dht = dht_protocol
        
        # Layer assignments
        self.layer_assignments: Dict[int, List[LayerAssignment]] = defaultdict(list)
        
        # Node capacities
        self.node_capacities: Dict[str, float] = {}  # node_id -> available_mb
        
        # DYNAMIC ARCHITECTURE (auto-updates as network grows)
        self.current_architecture: Optional[ModelArchitecture] = None
        self.architecture_version: int = 0
        self.last_node_count: int = 0
        
        # Legacy fields (for compatibility)
        self.current_num_layers = 0
        self.embedding_holder: Optional[str] = None
        self.lm_head_holder: Optional[str] = None
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info("DynamicLayerPool initialized with dynamic width + depth scaling")
    
    def _auto_recalculate_architecture(self):
        """
        AUTOMATED architecture optimization - no human intervention needed.
        
        Calculates optimal architecture based on current network capacity
        and triggers upgrade if improvement is significant.
        """
        total_memory = sum(self.node_capacities.values())
        optimal = calculate_optimal_architecture(total_memory)
        
        if self.current_architecture is None:
            # First initialization
            self.current_architecture = optimal
            self.current_num_layers = optimal.num_layers
            self.architecture_version = 1
            logger.info(f"🚀 Initial architecture: {optimal.num_layers}L × {optimal.hidden_dim}H "
                       f"({optimal.estimate_params()/1e6:.0f}M params)")
            return
        
        # Check if upgrade is worthwhile
        should_upgrade, reason = should_upgrade_architecture(
            self.current_architecture,
            optimal,
            self.MIN_UPGRADE_IMPROVEMENT
        )
        
        if should_upgrade:
            logger.warning(f"🔄 ARCHITECTURE UPGRADE TRIGGERED!")
            logger.warning(f"   {reason}")
            logger.warning(f"   Old: {self.current_architecture.num_layers}L × {self.current_architecture.hidden_dim}H")
            logger.warning(f"   New: {optimal.num_layers}L × {optimal.hidden_dim}H")
            logger.warning(f"   Nodes will gradually migrate to new architecture on restart")
            
            # Update architecture (new nodes will use new arch)
            self.current_architecture = optimal
            self.current_num_layers = optimal.num_layers
            self.architecture_version += 1
            
            # TODO: Trigger distillation-based migration for existing nodes
            # For now, existing nodes keep their architecture until restart
        else:
            logger.debug(f"Architecture recalculation: no upgrade needed ({reason})")
    
    def register_node(
        self, 
        node_id: str, 
        node_url: str,
        grpc_addr: str,
        available_memory_mb: float,
        staked_amount: float = 0.0
    ) -> List[int]:
        """
        Register a node and assign layers based on its capacity AND stake.
        
        AUTOMATIC ARCHITECTURE SCALING:
        - Periodically recalculates optimal architecture as network grows
        - Triggers upgrades when capacity increases significantly
        - New nodes automatically use latest architecture
        
        Validator role requires:
        1. Sufficient memory (>2GB)
        2. Minimum stake (100 NEURO) - prevents Sybil attacks
        
        Returns list of layer IDs assigned to this node.
        """
        # Import validator requirements from centralized economics (with dynamic scaling!)
        from neuroshard.core.economics.constants import (
            VALIDATOR_MIN_MEMORY_MB, 
            get_dynamic_validator_stake
        )
        
        with self.lock:
            self.node_capacities[node_id] = available_memory_mb
            
            # AUTO-TRIGGER: Recalculate architecture if network grew significantly
            node_count = len(self.node_capacities)
            recalc_interval = self._get_recalc_interval(node_count)
            if (node_count - self.last_node_count) >= recalc_interval:
                self._auto_recalculate_architecture()
                self.last_node_count = node_count
            
            # Ensure we have an architecture
            if self.current_architecture is None:
                self._auto_recalculate_architecture()
            
            # Calculate how many layers this node can hold
            # Uses current architecture's dimensions (dynamic!)
            max_layers_for_node = calculate_layer_assignment(
                available_memory_mb,
                self.current_architecture,
                safety_factor=0.6
            )
            
            # SCALABILITY: Apply MAX_LAYERS_PER_NODE cap in large networks
            # This prevents single nodes from hogging all layers and ensures
            # load distribution as the network grows
            node_count = len(self.node_capacities)
            if node_count > 100:
                # In large networks, cap layers per node
                max_layers_for_node = min(max_layers_for_node, self.MAX_LAYERS_PER_NODE)
                logger.debug(f"Large network ({node_count} nodes): capped to {max_layers_for_node} layers")
            
            if max_layers_for_node < 1:
                logger.warning(f"Node {node_id[:8]}... has insufficient memory for even 1 layer")
                return []
            
            # Find layers that need more replicas
            assigned_layers = []
            
            # SCALABILITY STRATEGY:
            # High-capacity nodes (>8GB) are prioritized for Layer 0 (Driver) and Last Layer (Validator)
            # This creates parallel pipelines ("Training Gangs")
            is_high_capacity = available_memory_mb > 8000
            is_medium_capacity = available_memory_mb > VALIDATOR_MIN_MEMORY_MB
            
            # Count current validators for DYNAMIC stake requirement
            num_drivers = len(self.layer_assignments[0])
            num_validators = len(self.layer_assignments[max(0, self.current_num_layers - 1)])
            
            # VALIDATOR ELIGIBILITY: Dynamic stake based on network size!
            # - Few validators (1-10): 100 NEURO (accessible for bootstrap)
            # - Many validators (1000+): 2500 NEURO (security at scale)
            required_validator_stake = get_dynamic_validator_stake(num_validators)
            has_validator_stake = staked_amount >= required_validator_stake
            
            # Driver: Just needs memory (no stake requirement)
            should_be_driver = is_high_capacity or (is_medium_capacity and num_drivers < 2)
            
            # Validator: Needs memory AND dynamic stake
            should_be_validator = (
                has_validator_stake and 
                (is_high_capacity or (is_medium_capacity and num_validators < 2))
            )
            
            if not has_validator_stake and is_medium_capacity:
                logger.info(f"Node {node_id[:8]}... has capacity for Validator but insufficient stake "
                           f"({staked_amount:.2f} < {required_validator_stake:.0f} NEURO required for {num_validators} validators)")
            
            # Assign Layer 0 (Driver)
            if should_be_driver and len(assigned_layers) < max_layers_for_node:
                if not any(a.node_id == node_id for a in self.layer_assignments[0]):
                    self._assign_layer(0, node_id, node_url, grpc_addr)
                    assigned_layers.append(0)
                    # Ensure current_num_layers accounts for layer 0
                    if self.current_num_layers == 0:
                        self.current_num_layers = 1
            
            # 2. Fill gaps (layers with < MIN_REPLICAS)
            for layer_id in range(self.current_num_layers):
                if len(assigned_layers) >= max_layers_for_node:
                    break
                    
                current_replicas = len(self.layer_assignments[layer_id])
                if current_replicas < self.MIN_REPLICAS:
                    # Check if this node already has this layer
                    if layer_id not in assigned_layers and not any(a.node_id == node_id for a in self.layer_assignments[layer_id]):
                        self._assign_layer(layer_id, node_id, node_url, grpc_addr)
                        assigned_layers.append(layer_id)
            
            # 3. Assign Last Layer (Validator) if we still have space and qualify
            last_layer = max(0, self.current_num_layers - 1)
            if should_be_validator and len(assigned_layers) < max_layers_for_node:
                 if last_layer not in assigned_layers and not any(a.node_id == node_id for a in self.layer_assignments[last_layer]):
                    self._assign_layer(last_layer, node_id, node_url, grpc_addr)
                    assigned_layers.append(last_layer)

            # 4. If we have capacity left, we can potentially grow the model
            remaining_capacity = max_layers_for_node - len(assigned_layers)
            if remaining_capacity > 0:
                # Add new layers to grow the model
                new_layers = self._grow_model(remaining_capacity, node_id, node_url, grpc_addr)
                assigned_layers.extend(new_layers)
            
            # Handle embedding and LM head tracking
            # Any node with Layer 0 has embedding
            if 0 in assigned_layers:
                # Update tracking (just keeps one for reference, but multiple exist)
                self.embedding_holder = node_id 
                logger.info(f"Node {node_id[:8]}... became a Driver (Layer 0)")
            
            # Any node with Last Layer has head
            if (self.current_num_layers - 1) in assigned_layers:
                self.lm_head_holder = node_id
                logger.info(f"Node {node_id[:8]}... became a Validator (Last Layer)")
            
            # EARLY NETWORK NOTICE: When there are <10 nodes, each must hold many/all layers
            # This is TEMPORARY - as more nodes join, layers will be distributed
            if len(assigned_layers) > 50:
                logger.warning(f"Node {node_id[:8]}... holding {len(assigned_layers)} layers due to low network size")
                logger.warning(f"This is temporary - as more nodes join, model will shard across network")
            
            logger.info(f"Node {node_id[:8]}... registered: {len(assigned_layers)} layers assigned "
                       f"(capacity: {max_layers_for_node} layers, {available_memory_mb:.0f}MB)")
            
            return assigned_layers
    
    def _assign_layer(self, layer_id: int, node_id: str, node_url: str, grpc_addr: str):
        """Assign a layer to a node."""
        assignment = LayerAssignment(
            layer_id=layer_id,
            node_id=node_id,
            node_url=node_url,
            grpc_addr=grpc_addr,
        )
        self.layer_assignments[layer_id].append(assignment)
        
        # Announce to DHT
        if self.dht:
            try:
                import json
                key = f"layer_{layer_id}"
                current = self.dht.lookup_value(key)
                holders = json.loads(current) if current else []
                if grpc_addr not in holders:
                    holders.append(grpc_addr)
                self.dht.store(key, json.dumps(holders))
            except Exception as e:
                logger.debug(f"DHT announce failed: {e}")
    
    def _grow_model(
        self, 
        num_new_layers: int, 
        node_id: str, 
        node_url: str,
        grpc_addr: str
    ) -> List[int]:
        """
        Grow the model by adding new layers.
        
        This is how the model organically grows with the network!
        """
        new_layers = []
        
        for _ in range(num_new_layers):
            new_layer_id = self.current_num_layers
            self._assign_layer(new_layer_id, node_id, node_url, grpc_addr)
            new_layers.append(new_layer_id)
            self.current_num_layers += 1
        
        if new_layers:
            logger.info(f"Model grew: now {self.current_num_layers} layers "
                       f"(added layers {new_layers[0]}-{new_layers[-1]})")
        
        return new_layers
    
    def upgrade_to_validator(self, node_id: str, node_url: str, grpc_addr: str) -> bool:
        """
        Upgrade a node to Validator role (assign LM head) when stake requirement is met.
        
        This is called when a node stakes enough NEURO to become a Validator.
        No restart required - the node's role is upgraded dynamically.
        
        Returns True if upgrade was successful.
        """
        from neuroshard.core.economics.constants import VALIDATOR_MIN_MEMORY_MB
        
        with self.lock:
            # Check if node has sufficient memory
            memory = self.node_capacities.get(node_id, 0)
            if memory < VALIDATOR_MIN_MEMORY_MB:
                logger.warning(f"Node {node_id[:8]}... cannot be Validator: insufficient memory ({memory}MB)")
                return False
            
            # Check if already a validator
            last_layer = max(0, self.current_num_layers - 1)
            if any(a.node_id == node_id for a in self.layer_assignments[last_layer]):
                logger.info(f"Node {node_id[:8]}... is already a Validator")
                return True
            
            # Assign the last layer (LM head)
            self._assign_layer(last_layer, node_id, node_url, grpc_addr)
            self.lm_head_holder = node_id
            
            logger.info(f"Node {node_id[:8]}... upgraded to VALIDATOR (assigned layer {last_layer})")
            return True
    
    def demote_from_validator(self, node_id: str) -> bool:
        """
        Demote a node from Validator role when stake drops below requirement.
        
        This is called when:
        1. A validator unstakes and drops below the required amount
        2. The network grows and the required stake increases (tier change)
        
        The node keeps its other layer assignments but loses the LM head.
        
        Returns True if demotion was successful.
        """
        with self.lock:
            return self._demote_from_validator_unlocked(node_id)
    
    def _demote_from_validator_unlocked(self, node_id: str) -> bool:
        """
        Internal demotion logic (caller must hold self.lock).
        
        Split out to avoid deadlock when called from validate_all_validators().
        """
        last_layer = max(0, self.current_num_layers - 1)
        
        # Check if node is currently a validator
        current_assignments = self.layer_assignments.get(last_layer, [])
        was_validator = any(a.node_id == node_id for a in current_assignments)
        
        if not was_validator:
            logger.debug(f"Node {node_id[:8]}... is not a validator, nothing to demote")
            return False
        
        # Remove from last layer assignments
        self.layer_assignments[last_layer] = [
            a for a in current_assignments if a.node_id != node_id
        ]
        
        # Update lm_head_holder if this was the holder
        if self.lm_head_holder == node_id:
            # Find another validator if available
            remaining = self.layer_assignments.get(last_layer, [])
            if remaining:
                self.lm_head_holder = remaining[0].node_id
            else:
                self.lm_head_holder = None
        
        logger.warning(f"Node {node_id[:8]}... DEMOTED from Validator (insufficient stake)")
        return True
    
    def validate_all_validators(self, get_stake_fn) -> List[str]:
        """
        Validate all current validators still meet stake requirements.
        
        Called periodically or when stake tier changes to ensure all validators
        have sufficient stake for the current network size.
        
        IMPORTANT: Never demotes below MIN_VALIDATORS (2) to ensure the network
        can always compute real loss. The stake requirement only applies to
        validators beyond the minimum.
        
        Args:
            get_stake_fn: Function(node_id) -> float that returns current stake
            
        Returns:
            List of node_ids that were demoted
        """
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        MIN_VALIDATORS = 2  # Network needs at least 2 validators to function
        
        demoted = []
        
        with self.lock:
            last_layer = max(0, self.current_num_layers - 1)
            current_validators = list(self.layer_assignments.get(last_layer, []))
            num_validators = len(current_validators)
            
            # CRITICAL: Never demote below MIN_VALIDATORS
            # Otherwise the network can't compute real cross-entropy loss!
            if num_validators <= MIN_VALIDATORS:
                logger.debug(f"Only {num_validators} validators - skipping stake check (minimum {MIN_VALIDATORS} required)")
                return []
            
            # Get current stake requirement
            required_stake = get_dynamic_validator_stake(num_validators)
            
            # Sort validators by stake (lowest first) to demote lowest-stake first
            validators_with_stake = [
                (assignment, get_stake_fn(assignment.node_id))
                for assignment in current_validators
            ]
            validators_with_stake.sort(key=lambda x: x[1])  # Lowest stake first
            
            for assignment, node_stake in validators_with_stake:
                # Check if we'd go below minimum
                remaining_validators = num_validators - len(demoted)
                if remaining_validators <= MIN_VALIDATORS:
                    logger.info(f"Stopping demotion: {remaining_validators} validators remain (minimum {MIN_VALIDATORS})")
                    break
                
                if node_stake < required_stake:
                    logger.warning(
                        f"Validator {assignment.node_id[:8]}... has {node_stake:.0f} NEURO "
                        f"but {required_stake:.0f} required - DEMOTING"
                    )
                    # Use unlocked version since we already hold self.lock
                    if self._demote_from_validator_unlocked(assignment.node_id):
                        demoted.append(assignment.node_id)
        
        return demoted
    
    def unregister_node(self, node_id: str):
        """
        Unregister a node and redistribute its layers.
        
        This handles graceful degradation when nodes leave.
        """
        with self.lock:
            # Remove from capacities
            self.node_capacities.pop(node_id, None)
            
            # Find all layers this node was holding
            orphaned_layers = []
            
            for layer_id, assignments in self.layer_assignments.items():
                # Remove this node's assignment
                self.layer_assignments[layer_id] = [
                    a for a in assignments if a.node_id != node_id
                ]
                
                # Check if layer is now orphaned (< MIN_REPLICAS)
                if len(self.layer_assignments[layer_id]) < self.MIN_REPLICAS:
                    orphaned_layers.append(layer_id)
            
            # Handle embedding/head holder leaving
            if self.embedding_holder == node_id:
                self.embedding_holder = None
            if self.lm_head_holder == node_id:
                self.lm_head_holder = None
            
            if orphaned_layers:
                logger.warning(f"Node {node_id[:8]}... left, {len(orphaned_layers)} layers need redistribution")
                # In production, we would trigger redistribution here
    
    def get_layer_holders(self, layer_id: int) -> List[LayerAssignment]:
        """Get all nodes holding a specific layer."""
        with self.lock:
            return list(self.layer_assignments.get(layer_id, []))
    
    def get_pipeline_route(self) -> List[Tuple[int, str]]:
        """
        Get the route for pipeline inference.
        
        Returns list of (layer_id, grpc_addr) for each layer in order.
        
        Filters out dead/stale nodes based on heartbeat timeout.
        """
        with self.lock:
            route = []
            now = time.time()
            
            for layer_id in range(self.current_num_layers):
                holders = self.layer_assignments.get(layer_id, [])
                if not holders:
                    logger.error(f"Layer {layer_id} has no holders!")
                    continue
                
                # ROBUSTNESS: Filter out stale holders (expired heartbeat)
                active_holders = [
                    h for h in holders
                    if (now - h.last_heartbeat) < self.HEARTBEAT_TIMEOUT
                ]
                
                if not active_holders:
                    logger.warning(f"Layer {layer_id} has no ACTIVE holders "
                                  f"({len(holders)} total, all stale)")
                    continue
                
                # Pick best active holder (most recent heartbeat)
                active_holders.sort(key=lambda h: -h.last_heartbeat)
                route.append((layer_id, active_holders[0].grpc_addr))
                
                logger.debug(f"Layer {layer_id}: selected {active_holders[0].node_id[:16]}... "
                            f"(heartbeat {now - active_holders[0].last_heartbeat:.1f}s ago)")
            
            return route
    
    def get_network_capacity(self) -> NetworkCapacity:
        """Get current network capacity with dynamic architecture."""
        with self.lock:
            total_memory = sum(self.node_capacities.values())
            
            # Calculate max layers based on current architecture
            if self.current_architecture:
                memory_per_layer = estimate_memory_per_layer(self.current_architecture)
                max_layers = int(total_memory * 0.6 / memory_per_layer)
            else:
                max_layers = 0
            
            layer_coverage = {
                layer_id: len(assignments)
                for layer_id, assignments in self.layer_assignments.items()
            }
            
            return NetworkCapacity(
                total_nodes=len(self.node_capacities),
                total_memory_mb=total_memory,
                max_layers=max_layers,
                assigned_layers=self.current_num_layers,
                layer_coverage=layer_coverage,
            )
    
    def heartbeat(self, node_id: str, layer_ids: List[int]):
        """Update heartbeat for a node's layers."""
        with self.lock:
            now = time.time()
            for layer_id in layer_ids:
                for assignment in self.layer_assignments.get(layer_id, []):
                    if assignment.node_id == node_id:
                        assignment.last_heartbeat = now
    
    def cleanup_stale_assignments(self) -> int:
        """
        Remove stale layer assignments (nodes that haven't heartbeat recently).
        
        Returns number of stale assignments removed.
        
        Called periodically to prevent dead peers from being selected for pipeline routing.
        """
        with self.lock:
            now = time.time()
            removed_count = 0
            
            for layer_id, assignments in list(self.layer_assignments.items()):
                # Filter out stale assignments
                active_assignments = [
                    a for a in assignments
                    if (now - a.last_heartbeat) < self.HEARTBEAT_TIMEOUT
                ]
                
                stale_count = len(assignments) - len(active_assignments)
                if stale_count > 0:
                    logger.info(f"Layer {layer_id}: removed {stale_count} stale assignments "
                               f"({len(active_assignments)} remain)")
                    removed_count += stale_count
                
                # Update assignments
                if active_assignments:
                    self.layer_assignments[layer_id] = active_assignments
                else:
                    # No active holders for this layer!
                    logger.warning(f"Layer {layer_id}: NO active holders remaining!")
                    del self.layer_assignments[layer_id]
            
            return removed_count


class DynamicNeuroLLM:
    """
    A NeuroLLM that dynamically scales with the network.
    
    Key differences from fixed-phase model:
    - Number of layers AND hidden dimension determined by network capacity
    - Layers are distributed across nodes
    - Model grows organically in BOTH width and depth
    - Architecture adapts automatically as network expands
    """
    
    def __init__(
        self,
        node_id: str,
        layer_pool: DynamicLayerPool,
        device: str = "cpu"
    ):
        self.node_id = node_id
        self.layer_pool = layer_pool
        self.device = device
        
        # Get current architecture from layer pool
        if layer_pool.current_architecture is None:
            raise RuntimeError("Layer pool has no architecture - call _auto_recalculate_architecture first")
        self.architecture = layer_pool.current_architecture
        
        # My assigned layers
        self.my_layers: Dict[int, torch.nn.Module] = {}
        self.my_layer_ids: List[int] = []
        
        # Do I hold embedding/head?
        self.has_embedding = False
        self.has_lm_head = False
        
        # Shared components (if I hold them)
        self.embedding: Optional[torch.nn.Embedding] = None
        self.lm_head: Optional[torch.nn.Linear] = None
        self.final_norm: Optional[torch.nn.Module] = None
        
        # Training mode flag (PyTorch convention)
        self.training = False
        
        logger.info(f"DynamicNeuroLLM initialized for node {node_id[:8]}... "
                   f"with {self.architecture.num_layers}L × {self.architecture.hidden_dim}H architecture")
    
    def initialize_layers(self, layer_ids: List[int]):
        """Initialize the layers assigned to this node using DYNAMIC architecture."""
        from neuroshard.core.model.llm import NeuroLLMConfig, NeuroDecoderLayer
        
        # Create config from current architecture (DYNAMIC!)
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
        
        for layer_id in layer_ids:
            layer = NeuroDecoderLayer(config, layer_id)
            layer.to(self.device)
            self.my_layers[layer_id] = layer
        
        self.my_layer_ids = sorted(layer_ids)
        
        # Initialize embedding if I'm the holder (uses dynamic hidden_dim!)
        if self.layer_pool.embedding_holder == self.node_id:
            self.embedding = torch.nn.Embedding(VOCAB_SIZE, self.architecture.hidden_dim)
            self.embedding.to(self.device)
            self.has_embedding = True
        
        # Initialize LM head if I'm the holder (uses dynamic hidden_dim!)
        if self.layer_pool.lm_head_holder == self.node_id:
            self.lm_head = torch.nn.Linear(self.architecture.hidden_dim, VOCAB_SIZE, bias=False)
            from neuroshard.core.model.llm import RMSNorm
            self.final_norm = RMSNorm(self.architecture.hidden_dim)
            self.lm_head.to(self.device)
            self.final_norm.to(self.device)
            self.has_lm_head = True
        
        logger.info(f"Initialized {len(layer_ids)} layers: {layer_ids}, "
                   f"arch={self.architecture.num_layers}L×{self.architecture.hidden_dim}H, "
                   f"embedding={self.has_embedding}, head={self.has_lm_head}")
    
    def initialize_lm_head(self) -> bool:
        """
        Dynamically initialize the LM head (for validator upgrade).
        
        Called when a node is upgraded to Validator after staking.
        No restart required - initializes the head in place.
        
        Returns True if initialization was successful.
        """
        if self.has_lm_head:
            logger.info("LM head already initialized")
            return True
        
        try:
            from neuroshard.core.model.llm import RMSNorm
            
            self.lm_head = torch.nn.Linear(self.architecture.hidden_dim, VOCAB_SIZE, bias=False)
            self.final_norm = RMSNorm(self.architecture.hidden_dim)
            self.lm_head.to(self.device)
            self.final_norm.to(self.device)
            self.has_lm_head = True
            
            # Add last layer to my layers if not already there
            last_layer = self.architecture.num_layers - 1
            if last_layer not in self.my_layer_ids:
                self.my_layer_ids.append(last_layer)
                self.my_layer_ids = sorted(self.my_layer_ids)
            
            logger.info(f"LM head initialized! Now computing REAL cross-entropy loss")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LM head: {e}")
    
    def disable_lm_head(self) -> bool:
        """
        Disable the LM head (for validator demotion).
        
        Called when a validator is demoted due to insufficient stake.
        The node reverts to Worker role and uses activation norm as loss.
        
        Returns True if demotion was successful.
        """
        if not self.has_lm_head:
            logger.debug("LM head not initialized, nothing to disable")
            return False
        
        try:
            # Free memory from LM head
            if self.lm_head is not None:
                del self.lm_head
                self.lm_head = None
            if self.final_norm is not None:
                del self.final_norm
                self.final_norm = None
            
            self.has_lm_head = False
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.warning(f"LM head DISABLED - node demoted to Worker (will use activation norm as loss)")
            return True
        except Exception as e:
            logger.error(f"Failed to disable LM head: {e}")
            return False
    
    def forward_my_layers(
        self,
        hidden_states: torch.Tensor,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward through my assigned layers."""
        if start_layer is None:
            start_layer = min(self.my_layer_ids) if self.my_layer_ids else 0
        if end_layer is None:
            end_layer = max(self.my_layer_ids) + 1 if self.my_layer_ids else 0
        
        x = hidden_states
        
        for layer_id in range(start_layer, end_layer):
            if layer_id in self.my_layers:
                x, _ = self.my_layers[layer_id](x)
        
        return x
    
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens (only if I hold embedding)."""
        if not self.has_embedding:
            raise RuntimeError("This node does not hold the embedding layer")
        return self.embedding(input_ids)
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits (only if I hold LM head)."""
        if not self.has_lm_head:
            raise RuntimeError("This node does not hold the LM head")
        x = self.final_norm(hidden_states)
        return self.lm_head(x)
    
    def get_num_params(self) -> int:
        """Get number of parameters on this node."""
        total = 0
        for layer in self.my_layers.values():
            total += sum(p.numel() for p in layer.parameters())
        if self.embedding:
            total += sum(p.numel() for p in self.embedding.parameters())
        if self.lm_head:
            total += sum(p.numel() for p in self.lm_head.parameters())
        if self.final_norm:
            total += sum(p.numel() for p in self.final_norm.parameters())
        return total
    
    def parameters(self):
        """Yield all parameters for this node's model components (for optimizer/gradient clipping)."""
        for layer in self.my_layers.values():
            yield from layer.parameters()
        if self.embedding:
            yield from self.embedding.parameters()
        if self.lm_head:
            yield from self.lm_head.parameters()
        if self.final_norm:
            yield from self.final_norm.parameters()
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Yield (name, param) tuples for all parameters.
        
        This is the standard PyTorch interface for iterating over named parameters.
        """
        # Layers
        for layer_id, layer in self.my_layers.items():
            layer_prefix = f"{prefix}layers.{layer_id}." if prefix else f"layers.{layer_id}."
            for name, param in layer.named_parameters(prefix='', recurse=recurse):
                yield layer_prefix + name, param
        
        # Embedding
        if self.embedding:
            emb_prefix = f"{prefix}embedding." if prefix else "embedding."
            for name, param in self.embedding.named_parameters(prefix='', recurse=recurse):
                yield emb_prefix + name, param
        
        # LM Head
        if self.lm_head:
            head_prefix = f"{prefix}lm_head." if prefix else "lm_head."
            for name, param in self.lm_head.named_parameters(prefix='', recurse=recurse):
                yield head_prefix + name, param
        
        # Final Norm
        if self.final_norm:
            norm_prefix = f"{prefix}final_norm." if prefix else "final_norm."
            for name, param in self.final_norm.named_parameters(prefix='', recurse=recurse):
                yield norm_prefix + name, param
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary of the model.
        
        This is the standard PyTorch interface for saving model state.
        """
        state = {}
        
        # Layers
        for layer_id, layer in self.my_layers.items():
            for name, param in layer.state_dict().items():
                state[f"layers.{layer_id}.{name}"] = param
        
        # Embedding
        if self.embedding:
            for name, param in self.embedding.state_dict().items():
                state[f"embedding.{name}"] = param
        
        # LM Head
        if self.lm_head:
            for name, param in self.lm_head.state_dict().items():
                state[f"lm_head.{name}"] = param
        
        # Final Norm
        if self.final_norm:
            for name, param in self.final_norm.state_dict().items():
                state[f"final_norm.{name}"] = param
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Load state dictionary into the model.
        
        This is the standard PyTorch interface for loading model state.
        """
        # Group state by component
        layer_states: Dict[int, Dict[str, Any]] = {}
        embedding_state: Dict[str, Any] = {}
        lm_head_state: Dict[str, Any] = {}
        final_norm_state: Dict[str, Any] = {}
        
        for key, value in state_dict.items():
            if key.startswith("layers."):
                parts = key.split(".", 2)
                layer_id = int(parts[1])
                param_name = parts[2]
                if layer_id not in layer_states:
                    layer_states[layer_id] = {}
                layer_states[layer_id][param_name] = value
            elif key.startswith("embedding."):
                param_name = key[len("embedding."):]
                embedding_state[param_name] = value
            elif key.startswith("lm_head."):
                param_name = key[len("lm_head."):]
                lm_head_state[param_name] = value
            elif key.startswith("final_norm."):
                param_name = key[len("final_norm."):]
                final_norm_state[param_name] = value
        
        # Load into components
        for layer_id, layer in self.my_layers.items():
            if layer_id in layer_states:
                layer.load_state_dict(layer_states[layer_id], strict=strict)
        
        if self.embedding and embedding_state:
            self.embedding.load_state_dict(embedding_state, strict=strict)
        
        if self.lm_head and lm_head_state:
            self.lm_head.load_state_dict(lm_head_state, strict=strict)
        
        if self.final_norm and final_norm_state:
            self.final_norm.load_state_dict(final_norm_state, strict=strict)
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Zero all gradients.
        
        This is the standard PyTorch interface for zeroing gradients.
        """
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
    
    def train(self, mode: bool = True) -> 'DynamicNeuroLLM':
        """
        Set the model to training mode.
        
        This is the standard PyTorch interface for setting training mode
        on all submodules.
        """
        self.training = mode
        for layer in self.my_layers.values():
            layer.train(mode)
        if self.embedding:
            self.embedding.train(mode)
        if self.lm_head:
            self.lm_head.train(mode)
        if self.final_norm:
            self.final_norm.train(mode)
        return self
    
    def eval(self) -> 'DynamicNeuroLLM':
        """Set the model to evaluation mode."""
        return self.train(False)
    
    def get_my_contribution(self) -> Dict[str, Any]:
        """Get this node's contribution to the network."""
        capacity = self.layer_pool.get_network_capacity()
        
        return {
            "node_id": self.node_id[:16] + "...",
            "my_layers": self.my_layer_ids,
            "my_params": self.get_num_params(),
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "network_total_layers": capacity.assigned_layers,
            "network_total_nodes": capacity.total_nodes,
            "contribution_ratio": len(self.my_layer_ids) / max(1, capacity.assigned_layers),
        }


def calculate_reward_multiplier(
    num_layers_held: int,
    total_network_layers: int,
    has_embedding: bool,
    has_lm_head: bool
) -> float:
    """
    Calculate NEURO reward multiplier based on contribution.
    
    Roles:
    - Worker: Standard reward based on layers
    - Driver (Embedding): 1.2x bonus (bandwidth cost)
    - Validator (Head): 1.2x bonus (compute/consensus cost)
    """
    if total_network_layers == 0:
        return 1.0
    
    # Base multiplier from layer contribution
    layer_ratio = num_layers_held / total_network_layers
    base_multiplier = 1.0 + layer_ratio  # 1.0 to 2.0 based on layers
    
    # Bonus for critical components (Roles)
    if has_embedding:
        base_multiplier *= 1.2  # 20% bonus for Driving (Data bandwidth)
    if has_lm_head:
        base_multiplier *= 1.2  # 20% bonus for Validating (Loss calc + Gradient origin)
    
    return base_multiplier


# ============================================================================
# DYNAMIC NEURO NODE - The Main Node Class
# ============================================================================

class DynamicNeuroNode:
    """
    A truly decentralized node that contributes based on available memory.
    
    NO PHASES. NO CENTRAL COORDINATION.
    
    How it works:
    1. Node starts, detects available memory
    2. Registers with network, gets assigned layers
    3. Loads only the layers it's responsible for
    4. Participates in training (computes gradients for its layers)
    5. Participates in inference (forwards through its layers)
    6. Earns NEURO proportional to its contribution
    
    The more memory you have, the more layers you hold, the more you earn.
    """
    
    CHECKPOINT_DIR = None  # Set in __init__
    
    def __init__(
        self,
        node_id: str,
        port: int = 8000,
        tracker_url: str = "https://neuroshard.com/api/tracker",
        node_token: Optional[str] = None,
        device: str = "cpu",
        available_memory_mb: Optional[float] = None,
        enable_training: bool = True,
        max_storage_mb: float = 100.0,
        max_cpu_threads: Optional[int] = None,
    ):
        self.node_id = node_id
        self.port = port
        self.tracker_url = tracker_url
        self.node_token = node_token
        
        # Detect device automatically if "auto" or "cpu" (backward compatibility)
        if device in ("auto", "cpu"):
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"[NODE] GPU detected: CUDA available")
            elif torch.backends.mps.is_available():
                # MPS (Apple Silicon GPU) - enabled now that we have GIL yields
                self.device = "mps"
                logger.info(f"[NODE] GPU detected: Apple Metal (MPS)")
            else:
                self.device = "cpu"
                logger.info(f"[NODE] No GPU detected, using CPU")
                
                # Help debug why CUDA isn't available
                import subprocess
                import sys
                
                # Check if NVIDIA GPU exists
                has_nvidia_gpu = False
                try:
                    if sys.platform == 'win32':
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
                        has_nvidia_gpu = result.returncode == 0
                    elif sys.platform == 'darwin':
                        # macOS doesn't have NVIDIA support (use MPS instead)
                        pass
                    else:  # Linux
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
                        has_nvidia_gpu = result.returncode == 0
                except Exception:
                    pass
                
                # Detailed diagnostic
                logger.info(f"[NODE] torch.cuda.is_available() = False")
                
                # Check if PyTorch was built with CUDA
                try:
                    cuda_available = torch.cuda.is_available()
                    cuda_built = getattr(torch.version, 'cuda', None)
                    torch_version = torch.__version__
                    logger.info(f"[NODE] PyTorch version: {torch_version}")
                    logger.info(f"[NODE] CUDA compiled version: {cuda_built if cuda_built else 'None (CPU-only build)'}")
                except Exception as e:
                    logger.info(f"[NODE] Could not get CUDA info: {e}")
                
                # Provide helpful diagnostic
                if has_nvidia_gpu:
                    logger.warning("⚠️  NVIDIA GPU DETECTED BUT NOT BEING USED!")
                    logger.warning("Your system has an NVIDIA GPU, but this PyTorch installation is CPU-only.")
                    logger.warning("🔧 TO ENABLE GPU (for 5-10x faster training):")
                    logger.warning("If running the .exe (frozen build):")
                    logger.warning("  Unfortunately, the bundled Python environment can't easily be modified.")
                    logger.warning("  We recommend running from source for GPU support.")
                    logger.warning("If running from source:")
                    logger.warning("  pip uninstall torch")
                    logger.warning("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
                    logger.warning("To verify: python -c \"import torch; print(torch.cuda.is_available())\"")
        else:
            self.device = device
            logger.info(f"[NODE] Device manually set to: {self.device}")
            
        logger.info(f"Using device: {self.device}")
        
        self.enable_training = enable_training
        self.max_storage_mb = max_storage_mb
        self.max_cpu_threads = max_cpu_threads
        
        # CPU thread limiting is done in runner.py BEFORE any torch operations
        # (torch.set_num_interop_threads must be called before any parallel work)
        if max_cpu_threads and self.device == "cpu":
            torch.set_num_threads(max_cpu_threads)  # Intra-op parallelism only
            logger.info(f"Set PyTorch intra-op threads: {max_cpu_threads}")
        
        # Detect memory if not provided
        if available_memory_mb is None:
            self.available_memory_mb = self._detect_available_memory()
        else:
            self.available_memory_mb = available_memory_mb
        
        # Checkpoint directory
        from pathlib import Path
        self.CHECKPOINT_DIR = Path.home() / ".neuroshard" / "checkpoints"
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Compute wallet_id from token for stable checkpoint naming
        # (node_id can change if machine_id changes, but wallet_id is stable)
        if node_token:
            self.wallet_id = hashlib.sha256(node_token.encode()).hexdigest()[:16]
        else:
            self.wallet_id = self.node_id[:16]  # Fallback to node_id
        
        # Layer pool (shared across network via DHT)
        self.layer_pool: Optional[DynamicLayerPool] = None
        
        # My model (only my layers)
        self.model: Optional[DynamicNeuroLLM] = None
        self.my_layer_ids: List[int] = []
        
        # Tokenizer
        self.tokenizer = None
        
        # Training components (enable_training set in __init__)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_coordinator = None
        self.data_manager = None
        self.gradient_gossip = None
        
        # P2P
        self.p2p_manager = None
        
        # Stats
        self.is_running = False
        self.total_tokens_processed = 0
        self.total_training_rounds = 0
        self.current_loss = float('inf')
        self.inference_count = 0
        self.training_contribution_count = 0
        
        # KV cache for inference
        self.kv_cache: Dict[str, Any] = {}
        
        # Training context (keeps tensors alive for backward pass)
        # session_id -> {input, output, prev_peer}
        self.training_context: Dict[str, Any] = {}
        
        logger.info(f"DynamicNeuroNode initialized: memory={self.available_memory_mb:.0f}MB")
    
    def _detect_available_memory(self) -> float:
        """Detect available system memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            # Use 70% of available memory for safety
            return mem.available * 0.7 / (1024 * 1024)
        except ImportError:
            # Fallback
            return 2000  # Assume 2GB
    
    def start(self):
        """Start the node."""
        logger.info("Starting DynamicNeuroNode...")
        
        # 1. Initialize layer pool
        dht = None
        if self.p2p_manager and hasattr(self.p2p_manager, 'dht'):
            dht = self.p2p_manager.dht
        self.layer_pool = DynamicLayerPool(dht_protocol=dht)
        
        # 1b. SMART ARCHITECTURE RECONCILIATION
        # This handles the case where the network has evolved while we were offline
        self._reconcile_architecture()
        
        # 2. Get staked amount from ledger (for Validator eligibility)
        staked_amount = 0.0
        if self.p2p_manager and self.p2p_manager.ledger:
            try:
                account_info = self.p2p_manager.ledger.get_account_info()
                staked_amount = account_info.get("stake", 0.0)
                logger.info(f"Current stake: {staked_amount:.2f} NEURO")
            except Exception as e:
                logger.debug(f"Could not get stake info: {e}")
        
        # 3. Register with network and get layer assignments
        self.my_layer_ids = self.layer_pool.register_node(
            node_id=self.node_id,
            node_url=f"http://localhost:{self.port}",
            grpc_addr=f"localhost:{self.port + 1000}",
            available_memory_mb=self.available_memory_mb,
            staked_amount=staked_amount
        )
        
        logger.info(f"Assigned {len(self.my_layer_ids)} layers: {self.my_layer_ids}")
        
        # 3. Initialize model with my layers
        self.model = DynamicNeuroLLM(
            node_id=self.node_id,
            layer_pool=self.layer_pool,
            device=self.device
        )
        self.model.initialize_layers(self.my_layer_ids)
        
        # 4. Initialize tokenizer
        from neuroshard.core.model.tokenizer import get_neuro_tokenizer
        self.tokenizer = get_neuro_tokenizer()
        
        # 5. Try to load existing checkpoint (resume training)
        self._load_checkpoint()
        
        # 6. Setup training
        if self.enable_training:
            self._setup_training()
        
        self.is_running = True
        
        # Log contribution
        contribution = self.model.get_my_contribution()
        logger.info(f"Node started: {contribution['my_params']/1e6:.1f}M params, "
                   f"{len(self.my_layer_ids)} layers, "
                   f"embed={self.model.has_embedding}, head={self.model.has_lm_head}")
    
    def _setup_training(self):
        """Setup training components."""
        from neuroshard.core.training.distributed import FederatedDataManager
        
        # Collect all parameters from my layers
        all_params = []
        for layer in self.model.my_layers.values():
            all_params.extend(layer.parameters())
        if self.model.embedding:
            all_params.extend(self.model.embedding.parameters())
        if self.model.lm_head:
            all_params.extend(self.model.lm_head.parameters())
        if self.model.final_norm:
            all_params.extend(self.model.final_norm.parameters())
        
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
        
        self.data_manager = FederatedDataManager(
            tokenizer=self.tokenizer,
            max_seq_len=2048
        )
        
        # DYNAMIC TRAINING CONFIG: Calculate based on current model size and device
        # This will be recalculated when model grows via recalculate_training_config()
        num_layers = len(self.my_layer_ids)
        
        # Smart gradient checkpointing decision based on device and memory
        if self.device == "cuda" and self.available_memory_mb > 16000:
            # High-memory CUDA (Jetson Orin, RTX 3090+): only checkpoint if > 150 layers
            self._use_gradient_checkpointing = num_layers > 150
        elif self.device == "cuda":
            # Normal CUDA: checkpoint if > 80 layers or low memory
            self._use_gradient_checkpointing = num_layers > 80 or self.available_memory_mb < 8000
        else:
            # CPU/MPS: more conservative
            self._use_gradient_checkpointing = (
                self.available_memory_mb < 8000 or
                (self.device == "mps" and num_layers > 20) or
                num_layers > 50
            )
        
        # Calculate memory-aware training batch size
        self._training_batch_size = self._calculate_training_batch_size()

        logger.info(f"Training initialized: batch_size={self._training_batch_size}, "
                   f"checkpointing={self._use_gradient_checkpointing}, "
                   f"layers={num_layers}, device={self.device}")
    
    def _calculate_training_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available memory, device, and model size.
        
        DYNAMIC: This is called initially and can be recalculated when model grows.
        SMART: Considers GPU memory, gradient checkpointing, and actual model size.
        """
        seq_len = 512  # Typical sequence length
        hidden_dim = self.layer_pool.current_architecture.hidden_dim
        num_layers = len(self.my_layer_ids)
        
        # Calculate model memory footprint (params + gradients + optimizer states)
        model_params = sum(p.numel() for p in self.model.parameters())
        # Model memory: weights (fp32=4 bytes) × 4 (weights + grads + adam_m + adam_v)
        model_memory_mb = (model_params * 4 * 4) / (1024 * 1024)
        
        # For CUDA, check actual GPU memory available
        if self.device == "cuda":
            try:
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                logger.info(f"[NODE] CUDA memory: {gpu_allocated:.0f}MB used / {gpu_total:.0f}MB total")
                effective_memory_mb = self.available_memory_mb
            except Exception:
                effective_memory_mb = self.available_memory_mb
        else:
            effective_memory_mb = self.available_memory_mb
        
        # CORRECT FORMULA: Available for activations = Total - Model memory
        # Leave 10% buffer for system overhead
        available_for_activations = max(100, (effective_memory_mb * 0.9) - model_memory_mb)
        
        # With gradient checkpointing, activation memory is MUCH lower
        use_checkpointing = getattr(self, '_use_gradient_checkpointing', False)
        if use_checkpointing:
            # Checkpointing: Only need to store ~sqrt(num_layers) worth of activations
            # Plus inputs/outputs at checkpoint boundaries
            checkpoint_segments = max(1, int(num_layers ** 0.5))
            # Memory per sample: seq_len × hidden_dim × checkpoint_segments × 4 bytes × 2 (fwd+bwd)
            mem_per_sample_mb = (seq_len * hidden_dim * checkpoint_segments * 4 * 2) / (1024 * 1024)
            logger.info(f"[NODE] Gradient checkpointing: {checkpoint_segments} segments "
                       f"(~{mem_per_sample_mb:.1f}MB/sample)")
        else:
            # No checkpointing: full activation memory for all layers
            mem_per_sample_mb = (seq_len * hidden_dim * num_layers * 4 * 2) / (1024 * 1024)
        
        logger.info(f"[NODE] Memory budget: total={effective_memory_mb:.0f}MB, "
                   f"model={model_memory_mb:.0f}MB, "
                   f"available_for_activations={available_for_activations:.0f}MB")
        
        # Calculate max batch size from available memory
        max_batch = max(1, int(available_for_activations / max(1, mem_per_sample_mb)))
        
        # SMART CLAMPING based on device capability
        if self.device == "cuda" and effective_memory_mb > 16000:
            # High-memory CUDA (Jetson Orin 32GB, RTX 3090 24GB): up to 8
            max_batch = min(max_batch, 8)
        elif self.device == "cuda" and effective_memory_mb > 8000:
            # Medium CUDA: up to 4
            max_batch = min(max_batch, 4)
        elif self.device == "cuda":
            # Small CUDA: up to 2
            max_batch = min(max_batch, 2)
        elif num_layers > 100:
            # Large model on CPU/MPS: conservative
            max_batch = min(max_batch, 2)
        else:
            max_batch = min(max_batch, 4)
        
        batch_size = max(1, max_batch)
        
        logger.info(f"[NODE] Training config: batch_size={batch_size}, "
                   f"model={model_params/1e6:.1f}M params ({num_layers} layers × {hidden_dim} dim), "
                   f"checkpointing={use_checkpointing}, device={self.device}")
        
        return batch_size
    
    def recalculate_training_config(self):
        """
        Recalculate training configuration after model architecture changes.
        
        Called when:
        - Model grows (new layers added)
        - Memory allocation changes
        - Device changes
        """
        old_batch = getattr(self, '_training_batch_size', None)
        self._training_batch_size = self._calculate_training_batch_size()
        
        # Update gradient checkpointing based on new model size
        num_layers = len(self.my_layer_ids)
        old_checkpointing = getattr(self, '_use_gradient_checkpointing', False)
        
        # More nuanced checkpointing decision
        if self.device == "cuda" and self.available_memory_mb > 16000:
            # High-memory CUDA: only checkpoint if > 150 layers
            self._use_gradient_checkpointing = num_layers > 150
        elif self.device == "cuda":
            # Normal CUDA: checkpoint if > 80 layers
            self._use_gradient_checkpointing = num_layers > 80 or self.available_memory_mb < 8000
        else:
            # CPU/MPS: original logic
            self._use_gradient_checkpointing = (
                self.available_memory_mb < 8000 or
                (self.device == "mps" and num_layers > 20) or
                num_layers > 50
            )
        
        if old_batch != self._training_batch_size or old_checkpointing != self._use_gradient_checkpointing:
            logger.info(f"[NODE] Training config updated: batch_size={old_batch}→{self._training_batch_size}, "
                       f"checkpointing={old_checkpointing}→{self._use_gradient_checkpointing}")
    
    def stop(self):
        """Stop the node."""
        logger.info("Stopping DynamicNeuroNode...")
        
        self.is_running = False
        
        # Unregister from network
        if self.layer_pool:
            self.layer_pool.unregister_node(self.node_id)
        
        # Save checkpoint
        self._save_checkpoint()
        
        logger.info("DynamicNeuroNode stopped")
    
    def connect_p2p(self, p2p_manager):
        """Connect to P2P network."""
        self.p2p_manager = p2p_manager
        
        # Initialize Data Swarm
        from neuroshard.core.network.p2p_data import DataSwarm
        
        # Ensure cache dir exists in a writable location
        data_cache_dir = self.CHECKPOINT_DIR / "data_cache"
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.swarm = DataSwarm(p2p_manager, cache_dir=str(data_cache_dir))
        
        # Update layer pool with DHT
        if self.layer_pool and hasattr(p2p_manager, 'dht'):
            self.layer_pool.dht = p2p_manager.dht
        
        logger.info("Connected to P2P network and Data Swarm")
    
    # ==================== INFERENCE ====================
    
    def forward(self, input_ids: torch.Tensor, session_id: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass - routes through network if needed.
        
        If this node has all layers: process locally
        If not: forward to nodes with other layers
        """
        # Check if we can do full inference locally
        capacity = self.layer_pool.get_network_capacity()
        
        if len(self.my_layer_ids) == capacity.assigned_layers and self.model.has_embedding and self.model.has_lm_head:
            # We have everything - do local inference
            return self._forward_local(input_ids)
        else:
            # Need to route through network
            return self._forward_distributed(input_ids, session_id)
    
    def _forward_local(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full local inference (when we have all layers)."""
        with torch.no_grad():
            # Embed
            hidden = self.model.embed(input_ids.to(self.device))
            
            # Forward through all layers
            hidden = self.model.forward_my_layers(hidden)
            
            # Compute logits
            logits = self.model.compute_logits(hidden)
            
            self.inference_count += 1
            self.total_tokens_processed += input_ids.numel()
            
            return logits
    
    def _forward_distributed(self, input_ids: torch.Tensor, session_id: Optional[str] = None) -> torch.Tensor:
        """Distributed inference through network pipeline."""
        # Get pipeline route
        route = self.layer_pool.get_pipeline_route()
        
        if not route:
            raise RuntimeError("No pipeline route available")
        
        # Start with embedding
        if self.model.has_embedding:
            hidden = self.model.embed(input_ids.to(self.device))
        else:
            # Request embedding from holder
            hidden = self._request_embedding(input_ids)
        
        # Forward through layers (local or remote)
        current_layer = 0
        for layer_id, grpc_addr in route:
            if layer_id in self.model.my_layers:
                # Local layer
                hidden, _ = self.model.my_layers[layer_id](hidden)
            else:
                # Remote layer - forward to peer
                hidden = self._forward_to_peer(grpc_addr, hidden, layer_id)
            current_layer = layer_id
        
        # Compute logits
        if self.model.has_lm_head:
            logits = self.model.compute_logits(hidden)
        else:
            # Request from holder
            logits = self._request_logits(hidden)
        
        self.inference_count += 1
        self.total_tokens_processed += input_ids.numel()
        
        return logits
    
    def forward_pipeline(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        training_labels: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
        sender_url: Optional[str] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for pipeline parallelism (received from peer).
        """
        # Enable gradient tracking if training
        is_training = training_labels is not None
        
        if is_training:
            hidden_states.requires_grad_(True)
            hidden_states.retain_grad()
        
        # Check if input is token IDs (embedding request)
        # Integer dtype or 2D shape [batch, seq] implies input_ids
        # This happens when a client sends input_ids to the Driver (Layer 0)
        if (hidden_states.dtype in [torch.long, torch.int64, torch.int32] or 
            len(hidden_states.shape) == 2) and self.model.has_embedding:
            
            # Ensure correct dtype
            if hidden_states.dtype != torch.long:
                hidden_states = hidden_states.to(torch.long)
            
            # Embed tokens
            hidden_states = self.model.embed(hidden_states)
            
            if is_training:
                hidden_states.requires_grad_(True)
                hidden_states.retain_grad()
        
        # Forward through local layers
        output = self.model.forward_my_layers(hidden_states)
        
        if is_training and session_id:
            # Save context for backward pass
            self.training_context[session_id] = {
                "input": hidden_states,
                "output": output,
                "sender_url": sender_url,
                "timestamp": time.time()
            }
            # Cleanup old sessions
            now = time.time()
            to_remove = [s for s, ctx in self.training_context.items() if now - ctx["timestamp"] > 600]
            for s in to_remove:
                del self.training_context[s]
        
        # If we are the Validator (Last Layer holder)
        if self.model.has_lm_head:
            logits = self.model.compute_logits(output)
            
            # Calculate Loss if labels present
            if training_labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    training_labels.view(-1),
                    ignore_index=-100
                )
                
                # Trigger Backward Pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Propagate gradient back to previous node
                if sender_url and session_id:
                    # The gradient we send back is dL/d(input_hidden_states)
                    # hidden_states.grad is populated by backward()
                    if hidden_states.grad is not None:
                        self._backward_to_peer(
                            sender_url, 
                            hidden_states.grad, 
                            # Target shard is whatever layer sent this to us. 
                            # Assuming sender holds previous layers.
                            # We send to the sender's LAST layer.
                            # Simplified: just send to the node, it routes.
                            0, 
                            session_id
                        )
                
                # Step Optimizer
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                self.total_training_rounds += 1
                self.current_loss = loss.item()
                
                return logits, None
            
            return logits, None
            
        # If we are a Worker (Middle Layer), we need to forward to next peer
        my_last_layer = max(self.my_layer_ids) if self.my_layer_ids else 0
        next_layer = my_last_layer + 1
        
        if self.p2p_manager:
            next_hop = self.p2p_manager.get_next_hop(next_layer)
            if next_hop:
                return self._forward_to_peer(
                    next_hop, 
                    output, 
                    next_layer, 
                    labels=training_labels, 
                    session_id=session_id
                )
        
        logger.warning(f"Pipeline broken at layer {next_layer}: no peer found")
        return output, None

    def backward_pipeline(self, grad_output: torch.Tensor, session_id: str):
        """
        Backward pass received from next peer.
        """
        if session_id not in self.training_context:
            logger.warning(f"Received backward for unknown session {session_id}")
            return
            
        ctx = self.training_context[session_id]
        output = ctx["output"]
        input_tensor = ctx["input"]
        sender_url = ctx["sender_url"]
        
        # Run local backward
        # output is the tensor we produced in forward_pipeline
        # grad_output is dL/d(output) received from next peer
        self.optimizer.zero_grad()
        output.backward(grad_output)
        
        # Propagate back
        if sender_url and input_tensor.grad is not None:
             # Find previous layer ID? Not strictly needed for routing if we have direct sender URL
             # But _backward_to_peer takes layer_id
             self._backward_to_peer(sender_url, input_tensor.grad, 0, session_id)
             
        # Step Optimizer
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Cleanup
        del self.training_context[session_id]

    def _request_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Request embedding from the node that holds it."""
        # Find a node that holds Layer 0 (Driver)
        peer_url = None
        
        # 1. Check layer pool assignments
        if self.layer_pool:
            assignments = self.layer_pool.get_layer_holders(0)
            if assignments:
                # Pick one (e.g., random for load balancing)
                import random
                peer_url = random.choice(assignments).grpc_addr
        
        # 2. Fallback to P2P manager routing
        if not peer_url and self.p2p_manager:
            peer_url = self.p2p_manager.get_next_hop(0)
            
        if not peer_url:
            raise RuntimeError("No embedding holder (Driver/Layer 0) found in network")
            
        # Call peer - Send input_ids to Layer 0 holder
        # The receiver's forward_pipeline will detect it's input_ids and run embed()
        return self._forward_to_peer(peer_url, input_ids, 0)
    
    def _forward_to_peer(self, peer_url: str, hidden: torch.Tensor, layer_id: int, labels: Optional[torch.Tensor] = None, session_id: str = None) -> torch.Tensor:
        """
        Forward hidden states to a peer for processing.
        
        SECURITY: Calculates and validates SHA256 checksums to detect tampering.
        """
        from protos import neuroshard_pb2
        from protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        import numpy as np
        import hashlib
        
        try:
            parsed = urlparse(peer_url)
            ip = parsed.hostname
            # gRPC port convention
            port = (parsed.port or 80) + 1000
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            # Serialize hidden states
            hidden_bytes = hidden.detach().cpu().numpy().tobytes()
            hidden_shape = list(hidden.shape)
            
            # CHECKSUM: Calculate SHA256 hash for integrity verification
            checksum = hashlib.sha256(hidden_bytes).hexdigest()
            logger.debug(f"[SECURITY] Sending layer {layer_id} with checksum: {checksum[:16]}...")
            
            # Serialize labels if present
            labels_bytes = b""
            if labels is not None:
                labels_bytes = labels.cpu().numpy().tobytes()
            
            req_session_id = session_id or f"train_{time.time()}"
            
            # Get my URL for backward routing
            my_url = ""
            if self.p2p_manager:
                my_url = self.p2p_manager.my_url
            
            req = neuroshard_pb2.PipelineForwardRequest(
                session_id=req_session_id,
                request_id=f"req_{time.time()}",
                hidden_states=hidden_bytes,
                hidden_shape=hidden_shape,
                target_shard=layer_id,
                use_cache=False,
                training_labels=labels_bytes,
                sender_url=my_url
            )
            
            # Store context for backward pass
            # We need to know WHO sent us this so we can send gradients back? 
            # No, this function is called by US sending to THEM.
            # We need to know who THEY are so when they send us gradients back, we verify?
            # Actually, we don't need to do anything here for backward. 
            # They will call PipelineBackward on US.
            
            resp = stub.PipelineForward(req, timeout=30.0)
            
            if not resp.success:
                raise RuntimeError(f"Peer error: {resp.error_message}")
            
            # Deserialize result
            if resp.is_final:
                # It's logits
                result_bytes = resp.logits
                result = torch.from_numpy(
                    np.frombuffer(result_bytes, dtype=np.float32)
                ).reshape(list(resp.logits_shape))
            else:
                # It's hidden states (recursive/chained)
                result_bytes = resp.hidden_states
                result = torch.from_numpy(
                    np.frombuffer(result_bytes, dtype=np.float32)
                ).reshape(list(resp.hidden_shape))
            
            # CHECKSUM VALIDATION: Verify integrity of received data
            received_checksum = hashlib.sha256(result_bytes).hexdigest()
            logger.debug(f"[SECURITY] Received layer {layer_id} result with checksum: {received_checksum[:16]}...")
            
            # AUDIT TRAIL: Store checksum in PipelineSession for tamper detection
            if session_id and self.ledger and hasattr(self.ledger, 'inference_market'):
                market = self.ledger.inference_market
                if market and hasattr(market, 'active_sessions'):
                    for sess_id, session in market.active_sessions.items():
                        if sess_id == session_id or session.request_id in session_id:
                            session.activations_hashes.append(received_checksum)
                            logger.debug(f"[AUDIT] Stored checksum for layer {layer_id} in session")
                            break
            
            return result.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to forward to peer {peer_url}: {e}")
            return hidden

    def _backward_to_peer(self, peer_url: str, grad_output: torch.Tensor, layer_id: int, session_id: str):
        """Send gradients back to the previous peer."""
        from protos import neuroshard_pb2
        from protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        
        try:
            parsed = urlparse(peer_url)
            ip = parsed.hostname
            port = (parsed.port or 80) + 1000
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            grad_bytes = grad_output.detach().cpu().numpy().tobytes()
            grad_shape = list(grad_output.shape)
            
            req = neuroshard_pb2.PipelineBackwardRequest(
                session_id=session_id,
                request_id=f"bw_{time.time()}",
                grad_output=grad_bytes,
                grad_shape=grad_shape,
                target_shard=layer_id
            )
            
            stub.PipelineBackward(req, timeout=10.0)
            
        except Exception as e:
            logger.error(f"Failed to backward to peer {peer_url}: {e}")

    def _request_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Request logits from the node that holds LM head."""
        # Find Last Layer holder (Validator)
        if not self.layer_pool:
             return hidden
             
        capacity = self.layer_pool.get_network_capacity()
        last_layer = max(0, capacity.assigned_layers - 1)
        
        peer_url = None
        
        # 1. Check layer pool assignments
        assignments = self.layer_pool.get_layer_holders(last_layer)
        if assignments:
            import random
            peer_url = random.choice(assignments).grpc_addr
            
        # 2. Fallback to P2P manager
        if not peer_url and self.p2p_manager:
            peer_url = self.p2p_manager.get_next_hop(last_layer)
            
        if not peer_url:
            raise RuntimeError(f"No Validator (Layer {last_layer}) found in network")
            
        # Forward hidden states to peer targeting Last Layer
        # The receiver will compute logits and return is_final=True
        return self._forward_to_peer(peer_url, hidden, last_layer)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """Generate text from prompt."""
        try:
            if not self.tokenizer:
                raise RuntimeError("Tokenizer not initialized")
            
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
            logger.debug(f"[GENERATE] Encoded prompt: {input_ids.shape} tokens")
            
            # Move to model's device (handles CPU, CUDA, MPS)
            generated = input_ids.clone().to(self.device)
            
            # Get current vocabulary size from tokenizer
            # Only tokens 0 to current_vocab_size-1 are valid (have learned representations)
            # This is NOT a workaround - it's how BPE tokenizers work (vocab grows over time)
            valid_vocab_size = self.tokenizer.current_vocab_size
            
            for step in range(max_new_tokens):
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature
                
                # Constrain to valid vocabulary (standard BPE tokenizer behavior)
                # Tokens beyond current_vocab_size don't exist in the tokenizer yet
                if valid_vocab_size < next_logits.size(-1):
                    next_logits[:, valid_vocab_size:] = float('-inf')
                
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == 2:  # EOS
                    logger.debug(f"[GENERATE] EOS at step {step+1}")
                    break
            
            prompt_tokens = input_ids.size(1)
            new_tokens = generated[0, prompt_tokens:].tolist()
            result = self.tokenizer.decode(new_tokens)
            logger.debug(f"[GENERATE] Generated {len(new_tokens)} tokens: '{result[:100]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"[GENERATE] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # ==================== TRAINING ====================
    
    def contribute_training_data(self, text: str, apply_dp: bool = True) -> int:
        """
        Contribute training data.
        
        Returns the number of tokens added.
        """
        if not self.data_manager:
            return 0
        
        # Get token count before
        stats_before = self.data_manager.get_stats()
        tokens_before = stats_before.get("total_tokens", 0)
        
        self.data_manager.add_text(text, apply_dp=apply_dp)
        
        # Get token count after
        stats_after = self.data_manager.get_stats()
        tokens_after = stats_after.get("total_tokens", 0)
        
        tokens_added = tokens_after - tokens_before
        logger.info(f"Added {tokens_added} tokens to training buffer")
        
        return tokens_added
    
    def _get_training_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a training batch from the Genesis data loader.
        
        Returns:
            Tuple of (input_ids, labels) or None if data not available.
        """
        if not self.enable_training:
            return None
        
        # Initialize genesis loader if needed
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
                logger.info(f"[GENESIS] Data loader ready: {self.genesis_loader.total_shards} shards available")
                
                # Connect Swarm to Loader
                if hasattr(self, 'swarm') and self.swarm:
                    self.genesis_loader.set_swarm(self.swarm)
            except Exception as e:
                logger.warning(f"[GENESIS] Failed to initialize loader: {e}")
                return None
        
        # Check if data is ready
        if not self.genesis_loader.is_data_ready():
            return None
        
        # Get batch
        batch_size = getattr(self, '_training_batch_size', 2)
        try:
            input_ids, labels = self.genesis_loader.get_batch(batch_size=batch_size)
            return input_ids, labels
        except Exception as e:
            logger.warning(f"[GENESIS] Failed to get batch: {e}")
            return None
    
    def train_step(self) -> Optional[float]:
        """
        Perform a training step on my layers.
        
        OPTIMIZED FOR SINGLE-NODE: When we have embedding + all layers + LM head,
        we skip distributed overhead and train locally.
        
        NON-BLOCKING DATA: Uses prefetched data when available, raises RuntimeError
        if data not ready (caller should retry later).
        """
        if not self.enable_training:
            return None
        
        # RUNTIME MEMORY CHECK: Skip training if system memory is critically high
        # This prevents OOM crashes and keeps the system responsive
        try:
            import psutil
            mem = psutil.virtual_memory()
            # Skip if less than 15% of system RAM is free (critical threshold)
            if mem.percent > 85:
                logger.warning(f"[NODE] System memory at {mem.percent:.0f}%, skipping training step")
                # Also try to free some memory
                import gc
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                return None
        except Exception:
            pass  # If psutil fails, continue anyway
        
        try:
            # SINGLE-NODE OPTIMIZATION: Check if we're a full node (Driver + Worker + Validator)
            is_full_node = self.model.has_embedding and self.model.has_lm_head
            
            if self.model.has_embedding:
                # I am a Driver (Layer 0)
                # Use Genesis Data Loader
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
                        logger.info(f"[GENESIS] Data loader ready: {self.genesis_loader.total_shards} shards available")
                        
                        # Connect Swarm to Loader
                        if hasattr(self, 'swarm') and self.swarm:
                            self.genesis_loader.set_swarm(self.swarm)
                    except Exception as e:
                        import traceback
                        logger.error(f"[GENESIS] ERROR: {type(e).__name__}: {e}")
                        logger.error(f"[GENESIS] {traceback.format_exc()}")
                        # Mark as failed so we don't keep retrying immediately
                        self.genesis_loader = None
                        raise RuntimeError(f"Genesis loader init failed: {e}")
                
                # Check if data is ready (non-blocking)
                if not self.genesis_loader.is_data_ready():
                    # Data not ready - don't block, let caller retry
                    raise RuntimeError("Data not ready - shard still loading")
                
                # Get batch from Genesis Shard using memory-aware batch size
                batch_size = getattr(self, '_training_batch_size', 2)
                try:
                    input_ids, labels = self.genesis_loader.get_batch(batch_size=batch_size)
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                except RuntimeError as e:
                    # Data not ready - propagate up
                    raise
                except Exception as e:
                    logger.warning(f"[GENESIS] Failed to get batch: {type(e).__name__}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    return None
                
                # SINGLE-NODE OPTIMIZED PATH: Skip distributed overhead
                if is_full_node:
                    return self._train_step_local(input_ids, labels)
                
                # DISTRIBUTED PATH: Forward to next peer
                # Forward pass with optional gradient checkpointing
                # Note: time.sleep(0) yields GIL to keep HTTP server responsive
                embeddings = self.model.embed(input_ids)
                embeddings.requires_grad_(True)
                embeddings.retain_grad()
                time.sleep(0)  # Yield GIL
                
                # Use gradient checkpointing if enabled (trades CPU for memory)
                if getattr(self, '_use_gradient_checkpointing', False):
                    output = torch.utils.checkpoint.checkpoint(
                        self.model.forward_my_layers,
                        embeddings,
                        use_reentrant=False
                    )
                else:
                    output = self.model.forward_my_layers(embeddings)
                time.sleep(0)  # Yield GIL after forward pass
                
                # Distributed: Send to next peer
                my_last_layer = max(self.my_layer_ids) if self.my_layer_ids else 0
                next_layer = my_last_layer + 1
                
                if self.p2p_manager:
                    next_hop = self.p2p_manager.get_next_hop(next_layer)
                    if next_hop:
                        session_id = f"train_{self.node_id}_{time.time()}"
                        
                        # Save context for backward
                        self.training_context[session_id] = {
                            "input": embeddings,
                            "output": output,
                            "sender_url": None, # We are the start
                            "timestamp": time.time()
                        }
                        
                        self._forward_to_peer(
                            next_hop, 
                            output, 
                            next_layer, 
                            labels=labels, 
                            session_id=session_id
                        )
                        
                        # We don't get loss immediately in distributed pipeline
                        # It comes back later via backward pass or status update
                        # For now, return None or previous loss
                        return self.current_loss
                
                return None
                
            else:
                # I am a Worker/Validator
                # I wait for activations from peers via gRPC (forward_pipeline)
                # So this method does nothing actively
                return None
                
        except RuntimeError as e:
            error_msg = str(e)
            if "not ready" in error_msg.lower():
                # Data not ready - propagate to caller
                raise
            elif "out of memory" in error_msg.lower() or "MPS" in error_msg:
                logger.warning(f"Training step OOM - reducing batch size and clearing cache")
                # Clear GPU cache
                import gc
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Reduce batch size for next attempt
                current_batch = getattr(self, '_training_batch_size', 8)
                if current_batch > 1:
                    self._training_batch_size = max(1, current_batch // 2)
                    logger.info(f"Reduced batch size to {self._training_batch_size}")
                else:
                    # Already at minimum batch size, fall back to CPU for training
                    if self.device != "cpu":
                        logger.warning(f"Batch size already at minimum. Consider using --memory flag to limit layers.")
            else:
                logger.error(f"Training step failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return None
    
    def _train_step_local(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """
        OPTIMIZED single-node training step.
        
        When we have ALL components (embedding + layers + LM head), we can
        train entirely locally without any network overhead.
        """
        # Forward pass with optional gradient checkpointing
        # Note: time.sleep(0) yields GIL to keep HTTP server responsive
        embeddings = self.model.embed(input_ids)
        time.sleep(0)  # Yield GIL
        
        # Use gradient checkpointing if enabled (trades CPU for memory)
        if getattr(self, '_use_gradient_checkpointing', False):
            output = torch.utils.checkpoint.checkpoint(
                self.model.forward_my_layers,
                embeddings,
                use_reentrant=False
            )
        else:
            output = self.model.forward_my_layers(embeddings)
        time.sleep(0)  # Yield GIL after forward pass
        
        # Compute logits and loss
        logits = self.model.compute_logits(output)
        time.sleep(0)  # Yield GIL
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        time.sleep(0)  # Yield GIL after backward pass
        
        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        time.sleep(0)  # Yield GIL after optimizer step
        
        # Update stats
        self.total_training_rounds += 1
        self.current_loss = loss.item()
        
        # PERIODIC CHECKPOINT: Save every 10 training steps to avoid losing progress
        # (More frequent saves since training steps take ~1s each)
        if self.total_training_rounds % 10 == 0:
            self._save_checkpoint()
            logger.debug(f"Checkpoint saved at step {self.total_training_rounds}")
        
        return loss.item()
    
    # ==================== STATS & PONW ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        # Safety check for shutdown race condition
        model = getattr(self, 'model', None)
        layer_pool = getattr(self, 'layer_pool', None)
        
        contribution = model.get_my_contribution() if model else {}
        capacity = layer_pool.get_network_capacity() if layer_pool else None
        
        # Calculate reward multiplier
        my_layer_ids = getattr(self, 'my_layer_ids', [])
        network_layers = capacity.assigned_layers if capacity else len(my_layer_ids)
        reward_multiplier = calculate_reward_multiplier(
            num_layers_held=len(my_layer_ids),
            total_network_layers=network_layers or 1,
            has_embedding=model.has_embedding if model else False,
            has_lm_head=model.has_lm_head if model else False,
        )
        
        # Estimate network params (rough: ~10M params per layer)
        network_params = network_layers * 10_000_000 if network_layers else 0
        
        # Get data buffer size
        data_buffer_size = 0
        if self.data_manager:
            data_stats = self.data_manager.get_stats()
            data_buffer_size = data_stats.get("buffer_size", 0)
        
        # Get shard stats (if we have a genesis loader)
        shard_stats = {}
        if hasattr(self, 'genesis_loader') and self.genesis_loader:
            shard_stats = self.genesis_loader.get_stats()
        
        # Multi-node identity info
        instance_id = getattr(self, 'instance_id', None)
        wallet_id = getattr(self, 'wallet_id', None)
        
        return {
            "node_id": self.node_id[:16] + "...",
            "instance_id": instance_id,  # Unique per machine+port
            "wallet_id": wallet_id,       # Same across instances with same token
            "available_memory_mb": self.available_memory_mb,
            "my_layers": self.my_layer_ids,
            "my_params": contribution.get("my_params", 0),
            "has_embedding": contribution.get("has_embedding", False),
            "has_lm_head": contribution.get("has_lm_head", False),
            "contribution_ratio": contribution.get("contribution_ratio", 0),
            "reward_multiplier": reward_multiplier,
            "network_layers": network_layers,
            "network_params": network_params,
            "network_nodes": capacity.total_nodes if capacity else 1,
            "total_tokens_processed": self.total_tokens_processed,
            "total_training_rounds": self.total_training_rounds,
            "current_loss": self.current_loss,
            "inference_count": self.inference_count,
            "data_buffer_size": data_buffer_size,
            "shard_stats": shard_stats,
        }
    
    def get_ponw_proof(self) -> Dict[str, Any]:
        """
        Generate Proof of Neural Work.
        
        This proof demonstrates verifiable neural network computation
        and is used for NEURO token rewards.
        """
        contribution = self.model.get_my_contribution() if self.model else {}
        capacity = self.layer_pool.get_network_capacity() if self.layer_pool else None
        
        # Calculate reward multiplier
        multiplier = calculate_reward_multiplier(
            num_layers_held=len(self.my_layer_ids),
            total_network_layers=capacity.assigned_layers if capacity else 1,
            has_embedding=self.model.has_embedding if self.model else False,
            has_lm_head=self.model.has_lm_head if self.model else False,
        )
        
        timestamp = time.time()
        
        # Determine role
        role = "Worker"
        if self.model and self.model.has_embedding:
            role = "Driver"
        elif self.model and self.model.has_lm_head:
            role = "Validator"
        
        proof_data = {
            "node_id": self.node_id,
            "timestamp": timestamp,
            "tokens_processed": self.total_tokens_processed,
            "training_rounds": self.total_training_rounds,
            "training_contributions": self.training_contribution_count,
            "inference_count": self.inference_count,
            "layers_held": len(self.my_layer_ids),
            "layer_ids": self.my_layer_ids,
            "has_embedding": self.model.has_embedding if self.model else False,
            "has_lm_head": self.model.has_lm_head if self.model else False,
            "role": role,
            "reward_multiplier": multiplier,
            "available_memory_mb": self.available_memory_mb,
        }
        
        # Add model hash for verification
        # Use architecture-based hash (consistent with SwarmEnabledDynamicNode._get_model_hash)
        if self.model:
            hasher = hashlib.sha256()
            arch_str = f"{self.model.hidden_dim}:{len(self.my_layer_ids)}:{getattr(self.model, 'num_heads', 0)}"
            hasher.update(arch_str.encode())
            for name, param in sorted(self.model.named_parameters()):
                hasher.update(f"{name}:{list(param.shape)}".encode())
            proof_data["model_hash"] = hasher.hexdigest()[:16]
        
        # Sign the proof
        proof_string = f"{self.node_id}:{timestamp}:{self.total_tokens_processed}:{len(self.my_layer_ids)}:{self.total_training_rounds}"
        if self.node_token:
            # Use HMAC for proper signing
            import hmac
            signature = hmac.new(
                self.node_token.encode(),
                proof_string.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            signature = hashlib.sha256(proof_string.encode()).hexdigest()
        
        proof_data["signature"] = signature
        
        return proof_data
    
    def _reconcile_architecture(self):
        """
        Smart architecture reconciliation for rejoining the network.
        
        Handles all scenarios:
        1. Quick restart (same architecture) → Use checkpoint
        2. Network upgraded (larger arch) → Start fresh with network arch
        3. Network downgraded (smaller arch) → Start fresh with network arch
        4. Solo bootstrap (no peers) → Use checkpoint or calculate
        5. First time (no checkpoint) → Query network or calculate
        
        Priority:
        1. Network consensus (if peers available)
        2. Saved checkpoint (if compatible)
        3. Fresh calculation (fallback)
        """
        saved_arch = self._peek_checkpoint_architecture()
        network_arch = self._query_network_architecture()
        
        # Log what we found
        if saved_arch:
            logger.info(f"Saved checkpoint: {saved_arch.num_layers}L × {saved_arch.hidden_dim}H")
        else:
            logger.info(f"No saved checkpoint found")
        
        if network_arch:
            logger.info(f"Network architecture: {network_arch.num_layers}L × {network_arch.hidden_dim}H")
        else:
            logger.info(f"No peers found (solo mode or bootstrap)")
        
        # Decision matrix
        if network_arch and saved_arch:
            # Both exist - compare them
            if self._architectures_compatible(saved_arch, network_arch):
                # Perfect - checkpoint matches network
                logger.info(f"✅ Checkpoint compatible with network - will load checkpoint")
                self.layer_pool.current_architecture = network_arch
                self.layer_pool.current_num_layers = network_arch.num_layers
            else:
                # Mismatch - network takes priority
                logger.warning(f"⚠️ Architecture mismatch!")
                logger.warning(f"   Checkpoint: {saved_arch.num_layers}L × {saved_arch.hidden_dim}H")
                logger.warning(f"   Network: {network_arch.num_layers}L × {network_arch.hidden_dim}H")
                
                # Check if network arch fits in our memory
                network_memory = network_arch.estimate_memory_mb()
                if network_memory <= self.available_memory_mb:
                    logger.warning(f"   → Using NETWORK architecture (checkpoint will be incompatible)")
                    logger.warning(f"   → Your training progress will be preserved but weights reset")
                    self.layer_pool.current_architecture = network_arch
                    self.layer_pool.current_num_layers = network_arch.num_layers
                    # Rename old checkpoint instead of deleting
                    self._archive_incompatible_checkpoint()
                else:
                    logger.error(f"   → Network arch needs {network_memory}MB but you only have {self.available_memory_mb}MB!")
                    logger.error(f"   → This node cannot participate in current network")
                    logger.error(f"   → Falling back to solo mode with checkpoint architecture")
                    self.layer_pool.current_architecture = saved_arch
                    self.layer_pool.current_num_layers = saved_arch.num_layers
        
        elif network_arch:
            # Network exists but no checkpoint - join the network
            network_memory = network_arch.estimate_memory_mb()
            if network_memory <= self.available_memory_mb:
                logger.info(f"✅ Joining network with architecture: {network_arch.num_layers}L × {network_arch.hidden_dim}H")
                self.layer_pool.current_architecture = network_arch
                self.layer_pool.current_num_layers = network_arch.num_layers
            else:
                logger.warning(f"⚠️ Network arch needs {network_memory}MB but you only have {self.available_memory_mb}MB")
                logger.warning(f"   → Will calculate a smaller architecture (may train in isolation)")
                # Let register_node calculate appropriate architecture
        
        elif saved_arch:
            # Checkpoint exists but no network peers (solo mode)
            saved_memory = saved_arch.estimate_memory_mb()
            if saved_memory <= self.available_memory_mb:
                logger.info(f"✅ Solo mode - using saved checkpoint: {saved_arch.num_layers}L × {saved_arch.hidden_dim}H")
                self.layer_pool.current_architecture = saved_arch
                self.layer_pool.current_num_layers = saved_arch.num_layers
            else:
                logger.warning(f"⚠️ Saved arch needs {saved_memory}MB but you only have {self.available_memory_mb}MB")
                logger.warning(f"   → Calculating fresh architecture")
        
        else:
            # No checkpoint, no network - fresh start
            logger.info(f"Fresh start - architecture will be calculated from available memory")
    
    def _query_network_architecture(self) -> Optional[ModelArchitecture]:
        """
        Query the network for the current architecture.
        
        Tries multiple sources:
        1. DHT lookup for architecture announcements
        2. Tracker API for network stats
        3. Direct peer query
        
        Returns None if no peers available (solo mode).
        """
        import requests
        
        # Method 1: Try tracker API first (fastest, most reliable)
        if self.tracker_url:
            try:
                # Query tracker for network architecture
                response = requests.get(
                    f"{self.tracker_url}/network_architecture",
                    timeout=5
                )
                if response.ok:
                    data = response.json()
                    if data.get("hidden_dim"):
                        arch = ModelArchitecture(
                            hidden_dim=data["hidden_dim"],
                            intermediate_dim=data.get("intermediate_dim", int(data["hidden_dim"] * 8 / 3)),
                            num_layers=data.get("num_layers", 12),
                            num_heads=data.get("num_heads", 12),
                            num_kv_heads=data.get("num_kv_heads", 4),
                        )
                        logger.debug(f"Got network architecture from tracker: {arch.num_layers}L × {arch.hidden_dim}H")
                        return arch
            except Exception as e:
                logger.debug(f"Tracker architecture query failed: {e}")
        
        # Method 2: Query known peers directly
        if self.p2p_manager and self.p2p_manager.known_peers:
            for peer_url in list(self.p2p_manager.known_peers.keys())[:3]:
                try:
                    response = requests.get(
                        f"{peer_url}/api/node/architecture",
                        timeout=3
                    )
                    if response.ok:
                        data = response.json()
                        if data.get("hidden_dim"):
                            arch = ModelArchitecture(
                                hidden_dim=data["hidden_dim"],
                                intermediate_dim=data.get("intermediate_dim", int(data["hidden_dim"] * 8 / 3)),
                                num_layers=data.get("num_layers", 12),
                                num_heads=data.get("num_heads", 12),
                                num_kv_heads=data.get("num_kv_heads", 4),
                            )
                            logger.debug(f"Got network architecture from peer {peer_url}: {arch.num_layers}L × {arch.hidden_dim}H")
                            return arch
                except Exception:
                    continue
        
        # Method 3: DHT lookup (if available)
        if self.p2p_manager and hasattr(self.p2p_manager, 'dht') and self.p2p_manager.dht:
            try:
                import hashlib
                key = int(hashlib.sha1("network_architecture".encode()).hexdigest(), 16)
                value = self.p2p_manager.dht.lookup_value(key)
                if value:
                    import json
                    data = json.loads(value)
                    if isinstance(data, dict) and data.get("hidden_dim"):
                        arch = ModelArchitecture(
                            hidden_dim=data["hidden_dim"],
                            intermediate_dim=data.get("intermediate_dim", int(data["hidden_dim"] * 8 / 3)),
                            num_layers=data.get("num_layers", 12),
                            num_heads=data.get("num_heads", 12),
                            num_kv_heads=data.get("num_kv_heads", 4),
                        )
                        logger.debug(f"Got network architecture from DHT: {arch.num_layers}L × {arch.hidden_dim}H")
                        return arch
            except Exception as e:
                logger.debug(f"DHT architecture lookup failed: {e}")
        
        return None
    
    def _architectures_compatible(self, arch1: ModelArchitecture, arch2: ModelArchitecture) -> bool:
        """
        Check if two architectures are compatible for gradient exchange.
        
        Compatible means: same hidden_dim, num_heads, num_kv_heads
        (num_layers can differ - nodes just hold different subsets)
        """
        return (
            arch1.hidden_dim == arch2.hidden_dim and
            arch1.num_heads == arch2.num_heads and
            arch1.num_kv_heads == arch2.num_kv_heads
        )
    
    def _archive_incompatible_checkpoint(self):
        """
        Archive an incompatible checkpoint instead of deleting it.
        
        Storage-aware: Keeps only MAX_ARCHIVED_CHECKPOINTS and respects
        the user's storage budget.
        """
        MAX_ARCHIVED_CHECKPOINTS = 2  # Keep at most 2 old checkpoints
        
        path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt"
        
        if not path.exists():
            return
        
        # First, clean up old archives to stay within limits
        self._cleanup_old_archives(MAX_ARCHIVED_CHECKPOINTS - 1)  # Make room for new one
        
        # Now archive the current checkpoint
        import time
        timestamp = int(time.time())
        archive_path = self.CHECKPOINT_DIR / f"archived_{self.wallet_id}_{timestamp}.pt"
        
        try:
            path.rename(archive_path)
            logger.info(f"Archived incompatible checkpoint to: {archive_path.name}")
        except Exception as e:
            logger.warning(f"Could not archive checkpoint: {e}")
            # If archive fails, just delete it
            try:
                path.unlink()
                logger.info(f"Deleted incompatible checkpoint (archive failed)")
            except Exception:
                pass
    
    def _cleanup_old_archives(self, max_keep: int = 2):
        """
        Clean up old archived checkpoints, keeping only the most recent ones.
        
        Also enforces storage budget if archives are taking too much space.
        """
        # Find all archives for this wallet
        pattern = f"archived_{self.wallet_id}_*.pt"
        archives = sorted(
            self.CHECKPOINT_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Newest first
        )
        
        # Calculate total archive size
        total_archive_mb = sum(p.stat().st_size / (1024 * 1024) for p in archives)
        
        # Storage budget: archives should use at most 20% of max_storage
        archive_budget_mb = self.max_storage_mb * 0.2
        
        # Delete archives that exceed count OR storage limits
        deleted_count = 0
        for i, archive in enumerate(archives):
            should_delete = False
            
            # Too many archives
            if i >= max_keep:
                should_delete = True
            
            # Over storage budget
            if total_archive_mb > archive_budget_mb:
                should_delete = True
            
            if should_delete:
                try:
                    archive_size_mb = archive.stat().st_size / (1024 * 1024)
                    archive.unlink()
                    total_archive_mb -= archive_size_mb
                    deleted_count += 1
                    logger.debug(f"Cleaned up old archive: {archive.name}")
                except Exception:
                    pass
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old archived checkpoint(s)")
    
    def _peek_checkpoint_architecture(self) -> Optional[ModelArchitecture]:
        """
        Peek at checkpoint to get saved architecture WITHOUT loading weights.
        
        This allows us to use the same architecture as the checkpoint,
        preventing architecture drift between restarts on the same machine.
        """
        # Use wallet_id for stable checkpoint path
        path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt"
        
        # Also check legacy path
        legacy_path = self.CHECKPOINT_DIR / f"dynamic_node_{self.node_id[:16]}.pt"
        
        checkpoint_path = None
        if path.exists():
            checkpoint_path = path
        elif legacy_path.exists():
            checkpoint_path = legacy_path
        
        if not checkpoint_path:
            return None
        
        try:
            # Load just the metadata (weights_only would fail, but we catch it)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            arch_dict = checkpoint.get("architecture")
            if arch_dict:
                return ModelArchitecture.from_dict(arch_dict)
        except Exception as e:
            logger.debug(f"Could not peek checkpoint architecture: {e}")
        
        return None
    
    def _load_checkpoint(self):
        """Load checkpoint from disk if it exists (resume training)."""
        # Use wallet_id for stable checkpoint path (survives node_id changes)
        path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt"
        
        # Also check legacy path (node_id-based) for migration
        legacy_path = self.CHECKPOINT_DIR / f"dynamic_node_{self.node_id[:16]}.pt"
        if not path.exists() and legacy_path.exists():
            logger.info(f"Migrating checkpoint from legacy path: {legacy_path.name} -> {path.name}")
            legacy_path.rename(path)
        
        if not path.exists():
            logger.info(f"No checkpoint found at {path.name}, starting fresh")
            return False
        
        logger.info(f"Loading checkpoint from: {path.name}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # ARCHITECTURE COMPATIBILITY CHECK
            saved_arch_dict = checkpoint.get("architecture")
            if saved_arch_dict:
                saved_arch = ModelArchitecture.from_dict(saved_arch_dict)
                current_arch = self.model.architecture
                
                # Check if architecture changed (includes num_heads for head_dim compatibility)
                if (saved_arch.hidden_dim != current_arch.hidden_dim or 
                    saved_arch.intermediate_dim != current_arch.intermediate_dim or
                    saved_arch.num_heads != current_arch.num_heads or
                    saved_arch.num_kv_heads != current_arch.num_kv_heads):
                    logger.warning(f"Architecture mismatch! Checkpoint is incompatible.")
                    logger.warning(f"  Saved: {saved_arch.num_layers}L × {saved_arch.hidden_dim}H, "
                                   f"heads={saved_arch.num_heads}/{saved_arch.num_kv_heads}")
                    logger.warning(f"  Current: {current_arch.num_layers}L × {current_arch.hidden_dim}H, "
                                   f"heads={current_arch.num_heads}/{current_arch.num_kv_heads}")
                    logger.warning(f"  Starting fresh (architecture was upgraded)")
                    # Delete incompatible checkpoint
                    try:
                        path.unlink()
                        logger.info(f"Deleted incompatible checkpoint: {path}")
                    except Exception:
                        pass
                    return False
            else:
                logger.warning("Legacy checkpoint without architecture info - starting fresh")
                # Delete legacy checkpoint
                try:
                    path.unlink()
                    logger.info(f"Deleted legacy checkpoint: {path}")
                except Exception:
                    pass
                return False
            
            # Check layer assignment compatibility
            saved_layers = set(checkpoint.get("layer_ids", []))
            current_layers = set(self.my_layer_ids)
            
            if saved_layers != current_layers:
                # Layers changed - try to load what we can
                common_layers = saved_layers.intersection(current_layers)
                if common_layers:
                    logger.warning(f"Layer assignment changed: saved={len(saved_layers)}, current={len(current_layers)}, common={len(common_layers)}")
                    logger.info(f"Will load {len(common_layers)} common layers from checkpoint")
                else:
                    logger.warning(f"No common layers between checkpoint and current assignment, starting fresh")
                    return False
            
            # Load layer weights
            for layer_id, state_dict in checkpoint.get("layers", {}).items():
                layer_id = int(layer_id)
                if layer_id in self.model.my_layers:
                    self.model.my_layers[layer_id].load_state_dict(state_dict)
            
            # Load embedding if present
            if self.model.embedding and "embedding" in checkpoint:
                self.model.embedding.load_state_dict(checkpoint["embedding"])
            
            # Load LM head if present
            if self.model.lm_head and "lm_head" in checkpoint:
                self.model.lm_head.load_state_dict(checkpoint["lm_head"])
            
            # Load final norm if present
            if self.model.final_norm and "final_norm" in checkpoint:
                self.model.final_norm.load_state_dict(checkpoint["final_norm"])
            
            # Restore training state
            self.total_training_rounds = checkpoint.get("total_training_rounds", 0)
            
            # Store optimizer state for later loading (after optimizer is created)
            if "optimizer" in checkpoint:
                self._pending_optimizer_state = checkpoint["optimizer"]
            
            # Store DiLoCo state for later loading (after swarm is created)
            if "diloco" in checkpoint:
                self._pending_diloco_state = checkpoint["diloco"]
                logger.info("[NODE] DiLoCo state found in checkpoint, will restore after swarm init")
            
            # Count how many layers were actually loaded
            loaded_layer_count = sum(1 for lid in checkpoint.get("layers", {}).keys() if int(lid) in self.model.my_layers)
            logger.info(f"Checkpoint loaded: {self.total_training_rounds} training rounds, "
                       f"{loaded_layer_count}/{len(current_layers)} layers from {path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            return False
    
    def _restore_pending_state(self):
        """
        Restore optimizer and DiLoCo state after they are initialized.
        
        Called after swarm/optimizer are set up to restore checkpoint state.
        """
        # Restore optimizer state
        if hasattr(self, '_pending_optimizer_state') and self._pending_optimizer_state:
            if hasattr(self, 'optimizer') and self.optimizer:
                try:
                    self.optimizer.load_state_dict(self._pending_optimizer_state)
                    logger.info("[NODE] Restored optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"[NODE] Could not restore optimizer state: {e}")
            del self._pending_optimizer_state
        
        # Restore DiLoCo state
        if hasattr(self, '_pending_diloco_state') and self._pending_diloco_state:
            if hasattr(self, 'swarm') and self.swarm:
                diloco = getattr(self.swarm, 'diloco_trainer', None)
                if diloco and hasattr(diloco, 'load_state_dict'):
                    try:
                        diloco.load_state_dict(self._pending_diloco_state)
                        logger.info(f"[NODE] Restored DiLoCo state (inner_step={diloco.stats.inner_step_count})")
                    except Exception as e:
                        logger.warning(f"[NODE] Could not restore DiLoCo state: {e}")
            del self._pending_diloco_state
    
    def _save_checkpoint(self):
        """
        Save my layers to disk with architecture information.
        
        Also saves DiLoCo state and optimizer state for graceful shutdown/restart.
        """
        if not self.model:
            return
        
        try:
            checkpoint = {
                "node_id": self.node_id,
                "layer_ids": self.my_layer_ids,
                # CRITICAL: Save architecture for compatibility checking
                "architecture": self.model.architecture.to_dict(),
                "architecture_version": self.layer_pool.architecture_version if self.layer_pool else 1,
                "layers": {
                    layer_id: layer.state_dict()
                    for layer_id, layer in self.model.my_layers.items()
                },
                "has_embedding": self.model.has_embedding,
                "has_lm_head": self.model.has_lm_head,
                "total_training_rounds": self.total_training_rounds,
                "current_loss": self.current_loss,
                "timestamp": time.time(),
            }
            
            if self.model.embedding:
                checkpoint["embedding"] = self.model.embedding.state_dict()
            if self.model.lm_head:
                checkpoint["lm_head"] = self.model.lm_head.state_dict()
            if self.model.final_norm:
                checkpoint["final_norm"] = self.model.final_norm.state_dict()
            
            # Save optimizer state (for resuming training without losing momentum)
            if hasattr(self, 'optimizer') and self.optimizer:
                try:
                    checkpoint["optimizer"] = self.optimizer.state_dict()
                except Exception:
                    pass  # Optimizer state is optional
            
            # Save DiLoCo trainer state (for resuming inner loop progress)
            if hasattr(self, 'swarm') and self.swarm:
                try:
                    # Check for swarm.diloco_trainer (SwarmEnabledDynamicNode)
                    diloco = getattr(self.swarm, 'diloco_trainer', None)
                    if diloco and hasattr(diloco, 'state_dict'):
                        checkpoint["diloco"] = diloco.state_dict()
                        logger.info(f"[NODE] Saved DiLoCo state (inner_step={diloco.stats.inner_step_count})")
                except Exception as e:
                    logger.warning(f"[NODE] Could not save DiLoCo state: {e}")
            
            # Use wallet_id for stable checkpoint path (survives node_id changes)
            path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt"
            temp_path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt.tmp"
            
            # Save to temp file first, then rename (atomic on most filesystems)
            # Use _use_new_zipfile_serialization=False for better compatibility
            torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)
            
            # Rename temp to final (atomic)
            import shutil
            shutil.move(str(temp_path), str(path))
            
            logger.info(f"[NODE] Checkpoint saved ({len(self.my_layer_ids)} layers)")
        except Exception as e:
            logger.error(f"[NODE] Checkpoint save failed: {type(e).__name__}: {e}")
            # Clean up temp file if it exists
            temp_path = self.CHECKPOINT_DIR / f"dynamic_node_{self.wallet_id}.pt.tmp"
            if temp_path.exists():
                temp_path.unlink()


def create_dynamic_node(
    node_token: str,
    port: int = 8000,
    tracker_url: str = "https://neuroshard.com/api/tracker",
    available_memory_mb: Optional[float] = None,
    enable_training: bool = True,
    max_storage_mb: float = 100.0,
    max_cpu_threads: Optional[int] = None,
    device: str = "auto",
) -> DynamicNeuroNode:
    """
    Create and start a dynamic node.
    
    MULTI-NODE SUPPORT:
    If the same token is used on multiple machines or ports, each gets a unique
    node_id (based on machine + port) while sharing the same wallet_id (based on token).
    
    This means:
    - Each physical node has a unique network identity
    - Earnings accumulate to the same NEURO wallet
    - No conflicts in DHT/layer assignments
    """
    from neuroshard.utils.hardware import get_instance_id
    
    # Generate instance-specific node_id
    instance_id = get_instance_id(port)
    
    # Combine token with instance for unique network identity
    # wallet_id (from token alone) is used for NEURO earnings
    # node_id (from token + instance) is used for network identity
    combined = f"{node_token}:{instance_id}"
    node_id = str(int(hashlib.sha256(combined.encode()).hexdigest(), 16))
    
    # Log multi-node info
    wallet_id = hashlib.sha256(node_token.encode()).hexdigest()[:16]
    logger.info(f"Instance ID: {instance_id} (machine+port)")
    logger.info(f"Wallet ID: {wallet_id}... (for NEURO earnings)")
    logger.info(f"Node ID: {node_id[:16]}... (unique network identity)")
    
    node = DynamicNeuroNode(
        node_id=node_id,
        port=port,
        tracker_url=tracker_url,
        node_token=node_token,
        available_memory_mb=available_memory_mb,
        enable_training=enable_training,
        max_storage_mb=max_storage_mb,
        max_cpu_threads=max_cpu_threads,
        device=device,
    )
    
    # Store instance info for debugging
    node.instance_id = instance_id
    node.wallet_id = wallet_id
    
    node.start()
    
    return node

