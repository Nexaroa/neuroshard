"""
Quorum System - Self-Organizing Training Groups

This module implements the quorum-based training system:
- Quorum: A group of speed-matched nodes that together hold a complete model
- QuorumRegistry: DHT-backed registry for quorum discovery
- QuorumLifecycle: State machine for quorum formation, operation, and dissolution

Key Concepts:
- A Quorum is a complete pipeline (covers all layers from embedding to LM head)
- Nodes in a quorum are speed-matched (within compatible tiers)
- Quorums train synchronously within, asynchronously across (via DiLoCo)
- Sessions last ~1 hour, with renewal checks at 80% of session time
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set

logger = logging.getLogger(__name__)


# =============================================================================
# QUORUM CONFIGURATION
# =============================================================================

# Session timing
BASE_SESSION_DURATION = 3600      # 1 hour default session
MAX_SESSION_DURATION = 14400      # 4 hours maximum
RENEWAL_CHECK_RATIO = 0.8         # Check at 80% of session
MIN_BATCHES_TO_RENEW = 1000       # Minimum batches for renewal eligibility

# Quorum formation
FORMATION_TIMEOUT = 60            # Seconds to wait for quorum formation
MIN_QUORUM_MEMBERS = 1            # Minimum members (adapts with network size)
MAX_QUORUM_MEMBERS = 16           # Maximum members per quorum

# Health monitoring
HEARTBEAT_INTERVAL = 30           # Seconds between heartbeats
STALE_THRESHOLD = 4               # Missed heartbeats before considered stale
OFFLINE_THRESHOLD = 120           # Seconds before marked offline (4 × 30)


# =============================================================================
# QUORUM LIFECYCLE STATES
# =============================================================================

class QuorumLifecycle(Enum):
    """State machine for quorum lifecycle."""
    FORMING = "forming"           # Gathering members
    ACTIVE = "active"             # Training in progress
    RENEWING = "renewing"         # Session renewal in progress
    DISSOLVING = "dissolving"     # Graceful shutdown
    DISSOLVED = "dissolved"       # Quorum no longer exists


class QuorumRole(Enum):
    """Role within a quorum pipeline."""
    INITIATOR = "initiator"       # Holds embedding layer, fetches batches
    PROCESSOR = "processor"       # Middle layers, forward/backward
    FINISHER = "finisher"         # Holds LM head, computes loss


# =============================================================================
# QUORUM MEMBER
# =============================================================================

@dataclass
class QuorumMember:
    """A member of a quorum."""
    node_id: str
    endpoint: str                  # gRPC endpoint (ip:port)
    layer_range: Tuple[int, int]   # (start, end) exclusive
    speed_tier: str                # SpeedTier value
    role: QuorumRole = QuorumRole.PROCESSOR
    
    # Health tracking
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    batches_processed: int = 0
    
    # Uptime tracking (for reputation)
    join_time: float = field(default_factory=time.time)
    total_online_seconds: float = 0.0
    total_expected_seconds: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Derived reputation (0.0 to 1.0)
    # Combines uptime_ratio and success_rate
    reputation: float = 1.0
    
    def update_uptime(self, heartbeat_received: bool = True):
        """
        Update uptime tracking based on heartbeat status.
        
        Called periodically (e.g., every heartbeat interval) to track
        whether the node was online.
        
        Args:
            heartbeat_received: Whether a heartbeat was received in this interval
        """
        self.total_expected_seconds += HEARTBEAT_INTERVAL
        if heartbeat_received:
            self.total_online_seconds += HEARTBEAT_INTERVAL
        self._recalculate_reputation()
    
    def record_request_result(self, success: bool):
        """
        Record the result of a request processed by this member.
        
        Args:
            success: Whether the request completed successfully
        """
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self._recalculate_reputation()
    
    def _recalculate_reputation(self):
        """
        Recalculate reputation based on uptime and success rate.
        
        Reputation formula:
        - 60% weighted by uptime_ratio
        - 40% weighted by success_rate
        - Minimum 0.1 (to allow recovery)
        """
        uptime_ratio = self.uptime_ratio
        success_rate = self.success_rate
        
        # Weighted combination
        self.reputation = max(0.1, 0.6 * uptime_ratio + 0.4 * success_rate)
    
    @property
    def uptime_ratio(self) -> float:
        """
        Get the uptime ratio (0.0 to 1.0).
        
        This is the fraction of expected time the node was actually online.
        """
        if self.total_expected_seconds <= 0:
            return 1.0  # New nodes start with full uptime
        return min(1.0, self.total_online_seconds / self.total_expected_seconds)
    
    @property
    def success_rate(self) -> float:
        """
        Get the request success rate (0.0 to 1.0).
        
        This is the fraction of requests that completed successfully.
        """
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0  # New nodes start with full success
        return self.successful_requests / total
    
    @property
    def meets_pipeline_requirements(self) -> bool:
        """
        Check if member meets requirements for pipeline mode.
        
        Pipeline mode requires:
        - Uptime ratio >= 90%
        - Not currently stale or offline
        """
        return self.uptime_ratio >= 0.9 and not self.is_stale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "layer_range": list(self.layer_range),
            "speed_tier": self.speed_tier,
            "role": self.role.value,
            "last_heartbeat": self.last_heartbeat,
            "batches_processed": self.batches_processed,
            "reputation": self.reputation,
            "join_time": self.join_time,
            "total_online_seconds": self.total_online_seconds,
            "total_expected_seconds": self.total_expected_seconds,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuorumMember':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            endpoint=data["endpoint"],
            layer_range=tuple(data["layer_range"]),
            speed_tier=data["speed_tier"],
            role=QuorumRole(data.get("role", "processor")),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            batches_processed=data.get("batches_processed", 0),
            reputation=data.get("reputation", 1.0),
            join_time=data.get("join_time", time.time()),
            total_online_seconds=data.get("total_online_seconds", 0.0),
            total_expected_seconds=data.get("total_expected_seconds", 0.0),
            successful_requests=data.get("successful_requests", 0),
            failed_requests=data.get("failed_requests", 0),
        )
    
    @property
    def is_stale(self) -> bool:
        """Check if member hasn't sent heartbeat recently."""
        return (time.time() - self.last_heartbeat) > (HEARTBEAT_INTERVAL * STALE_THRESHOLD)
    
    @property
    def is_offline(self) -> bool:
        """Check if member is considered offline."""
        return (time.time() - self.last_heartbeat) > OFFLINE_THRESHOLD


# =============================================================================
# QUORUM
# =============================================================================

@dataclass
class Quorum:
    """
    A self-organized group of speed-matched nodes forming a complete pipeline.
    
    Properties:
    - Complete: Covers all layers from embedding (L0) to LM head
    - Speed-matched: All members in compatible speed tiers
    - Self-sufficient: Can train independently without coordinator
    - Temporary: Sessions last ~1 hour, then renew or dissolve
    """
    quorum_id: str
    speed_tier: str                # Dominant speed tier (e.g., "tier2")
    
    # Members and layer mapping
    members: List[QuorumMember] = field(default_factory=list)
    
    # Session management
    session_start: float = field(default_factory=time.time)
    session_end: float = 0.0
    lifecycle: QuorumLifecycle = QuorumLifecycle.FORMING
    
    # Training progress
    total_batches: int = 0
    total_steps: int = 0
    current_step: int = 0
    
    # Network versioning
    arch_version: int = 1
    vocab_version: int = 1
    
    # Throughput tracking
    batches_per_second: float = 0.0
    last_throughput_update: float = 0.0
    
    def __post_init__(self):
        """Initialize session end time if not set."""
        if self.session_end == 0.0:
            self.session_end = self.session_start + BASE_SESSION_DURATION
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique quorum ID."""
        return f"q-{uuid.uuid4().hex[:12]}"
    
    def to_json(self) -> str:
        """Serialize to JSON for DHT storage."""
        data = {
            "quorum_id": self.quorum_id,
            "speed_tier": self.speed_tier,
            "members": [m.to_dict() for m in self.members],
            "session_start": self.session_start,
            "session_end": self.session_end,
            "lifecycle": self.lifecycle.value,
            "total_batches": self.total_batches,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "arch_version": self.arch_version,
            "vocab_version": self.vocab_version,
            "batches_per_second": self.batches_per_second,
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Quorum':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        quorum = cls(
            quorum_id=data["quorum_id"],
            speed_tier=data["speed_tier"],
            session_start=data["session_start"],
            session_end=data["session_end"],
            lifecycle=QuorumLifecycle(data["lifecycle"]),
            total_batches=data.get("total_batches", 0),
            total_steps=data.get("total_steps", 0),
            current_step=data.get("current_step", 0),
            arch_version=data.get("arch_version", 1),
            vocab_version=data.get("vocab_version", 1),
            batches_per_second=data.get("batches_per_second", 0.0),
        )
        quorum.members = [QuorumMember.from_dict(m) for m in data.get("members", [])]
        return quorum
    
    # =========================================================================
    # LAYER COVERAGE
    # =========================================================================
    
    @property
    def layer_coverage(self) -> Set[int]:
        """Get set of all layers covered by members."""
        layers = set()
        for member in self.members:
            for layer_id in range(member.layer_range[0], member.layer_range[1]):
                layers.add(layer_id)
        return layers
    
    @property
    def is_complete(self) -> bool:
        """Check if quorum covers all layers (complete pipeline)."""
        if not self.members:
            return False
        
        # Get layer range from members
        all_layers = self.layer_coverage
        if not all_layers:
            return False
        
        min_layer = min(all_layers)
        max_layer = max(all_layers)
        
        # Check for gaps
        expected = set(range(min_layer, max_layer + 1))
        return all_layers == expected
    
    @property
    def has_initiator(self) -> bool:
        """Check if quorum has a node holding layer 0 (embedding)."""
        return any(m.role == QuorumRole.INITIATOR for m in self.members)
    
    @property
    def has_finisher(self) -> bool:
        """Check if quorum has a node holding the last layer (LM head)."""
        return any(m.role == QuorumRole.FINISHER for m in self.members)
    
    # =========================================================================
    # MEMBER MANAGEMENT
    # =========================================================================
    
    def add_member(self, member: QuorumMember) -> bool:
        """
        Add a member to the quorum.
        
        Args:
            member: QuorumMember to add
            
        Returns:
            True if added successfully
        """
        if len(self.members) >= MAX_QUORUM_MEMBERS:
            return False
        
        # Check for duplicate
        if any(m.node_id == member.node_id for m in self.members):
            return False
        
        # Assign role based on layer range
        if member.layer_range[0] == 0:
            member.role = QuorumRole.INITIATOR
        elif not self.members:
            # First member with non-zero start might be finisher
            member.role = QuorumRole.PROCESSOR
        else:
            # Check if this is the highest layer
            current_max = max(m.layer_range[1] for m in self.members)
            if member.layer_range[1] > current_max:
                member.role = QuorumRole.FINISHER
                # Demote previous finisher
                for m in self.members:
                    if m.role == QuorumRole.FINISHER:
                        m.role = QuorumRole.PROCESSOR
            else:
                member.role = QuorumRole.PROCESSOR
        
        self.members.append(member)
        logger.info(f"Added member {member.node_id[:8]}... to quorum {self.quorum_id[:8]}... as {member.role.value}")
        return True
    
    def remove_member(self, node_id: str) -> Optional[QuorumMember]:
        """
        Remove a member from the quorum.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            Removed member if found, None otherwise
        """
        for i, member in enumerate(self.members):
            if member.node_id == node_id:
                removed = self.members.pop(i)
                logger.info(f"Removed member {node_id[:8]}... from quorum {self.quorum_id[:8]}...")
                return removed
        return None
    
    def get_member(self, node_id: str) -> Optional[QuorumMember]:
        """Get a member by node ID."""
        for member in self.members:
            if member.node_id == node_id:
                return member
        return None
    
    def get_pipeline_order(self) -> List[QuorumMember]:
        """Get members in pipeline order (by layer range start)."""
        return sorted(self.members, key=lambda m: m.layer_range[0])
    
    def get_next_hop(self, current_node_id: str) -> Optional[QuorumMember]:
        """Get the next node in the pipeline after current node."""
        pipeline = self.get_pipeline_order()
        for i, member in enumerate(pipeline):
            if member.node_id == current_node_id:
                if i + 1 < len(pipeline):
                    return pipeline[i + 1]
        return None
    
    def get_prev_hop(self, current_node_id: str) -> Optional[QuorumMember]:
        """Get the previous node in the pipeline before current node."""
        pipeline = self.get_pipeline_order()
        for i, member in enumerate(pipeline):
            if member.node_id == current_node_id:
                if i > 0:
                    return pipeline[i - 1]
        return None
    
    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================
    
    def update_heartbeat(self, node_id: str) -> bool:
        """
        Update heartbeat timestamp for a member.
        
        Args:
            node_id: ID of node sending heartbeat
            
        Returns:
            True if member found and updated
        """
        member = self.get_member(node_id)
        if member:
            member.last_heartbeat = time.time()
            member.consecutive_failures = 0
            return True
        return False
    
    def record_failure(self, node_id: str) -> int:
        """
        Record a failure for a member.
        
        Returns:
            Number of consecutive failures
        """
        member = self.get_member(node_id)
        if member:
            member.consecutive_failures += 1
            return member.consecutive_failures
        return 0
    
    def get_healthy_members(self) -> List[QuorumMember]:
        """Get list of members that are not stale."""
        return [m for m in self.members if not m.is_stale]
    
    def get_stale_members(self) -> List[QuorumMember]:
        """Get list of stale members (missed heartbeats)."""
        return [m for m in self.members if m.is_stale]
    
    def get_offline_members(self) -> List[QuorumMember]:
        """Get list of offline members."""
        return [m for m in self.members if m.is_offline]
    
    @property
    def is_healthy(self) -> bool:
        """Check if quorum has enough healthy members for operation."""
        healthy = self.get_healthy_members()
        return len(healthy) >= len(self.members) * 0.5 and self.is_complete
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    @property
    def session_remaining(self) -> float:
        """Get remaining session time in seconds."""
        return max(0, self.session_end - time.time())
    
    @property
    def session_progress(self) -> float:
        """Get session progress as fraction (0.0 to 1.0)."""
        duration = self.session_end - self.session_start
        elapsed = time.time() - self.session_start
        return min(1.0, elapsed / duration)
    
    @property
    def should_check_renewal(self) -> bool:
        """Check if it's time to check for renewal."""
        return self.session_progress >= RENEWAL_CHECK_RATIO
    
    @property
    def is_session_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() >= self.session_end
    
    def extend_session(self, duration: float = BASE_SESSION_DURATION) -> None:
        """Extend the session by specified duration."""
        self.session_end = time.time() + min(duration, MAX_SESSION_DURATION)
        self.lifecycle = QuorumLifecycle.ACTIVE
        logger.info(f"Quorum {self.quorum_id[:8]}... session extended by {duration}s")
    
    # =========================================================================
    # TRAINING PROGRESS
    # =========================================================================
    
    def record_batch(self, batches: int = 1) -> None:
        """Record processed batches."""
        self.total_batches += batches
        
        # Update throughput
        now = time.time()
        if self.last_throughput_update > 0:
            elapsed = now - self.last_throughput_update
            if elapsed > 0:
                # Exponential moving average
                new_rate = batches / elapsed
                self.batches_per_second = 0.9 * self.batches_per_second + 0.1 * new_rate
        self.last_throughput_update = now
    
    def record_step(self) -> None:
        """Record a training step completion."""
        self.total_steps += 1
        self.current_step += 1


# =============================================================================
# QUORUM REGISTRY
# =============================================================================

class QuorumRegistry:
    """
    DHT-backed registry for quorum discovery and management.
    
    Responsibilities:
    - Store quorum metadata in DHT
    - Discover existing quorums
    - Track local quorum membership
    """
    
    # DHT key prefixes
    DHT_KEY_QUORUM_PREFIX = "quorum:"
    DHT_KEY_QUORUM_LIST = "quorums:active"
    
    def __init__(self, dht_protocol=None):
        """
        Initialize the quorum registry.
        
        Args:
            dht_protocol: DHTProtocol instance for network storage
        """
        self.dht = dht_protocol
        self.local_quorums: Dict[str, Quorum] = {}  # quorum_id -> Quorum
        self._lock = threading.RLock()
    
    def _get_quorum_key(self, quorum_id: str) -> int:
        """Get DHT key for a quorum."""
        key_str = f"{self.DHT_KEY_QUORUM_PREFIX}{quorum_id}"
        return int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
    
    def register_quorum(self, quorum: Quorum) -> bool:
        """
        Register a quorum in the registry and DHT.
        
        Args:
            quorum: Quorum to register
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            self.local_quorums[quorum.quorum_id] = quorum
            
            # Store in DHT
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum.quorum_id)
                    self.dht.storage[key_id] = quorum.to_json()
                    
                    # Update active quorums list
                    self._update_active_list(quorum.quorum_id, add=True)
                    
                except Exception as e:
                    logger.warning(f"Failed to store quorum in DHT: {e}")
            
            logger.info(f"Registered quorum {quorum.quorum_id[:8]}... with {len(quorum.members)} members")
            return True
    
    def unregister_quorum(self, quorum_id: str) -> bool:
        """
        Unregister a quorum from the registry.
        
        Args:
            quorum_id: ID of quorum to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if quorum_id in self.local_quorums:
                del self.local_quorums[quorum_id]
            
            # Remove from DHT
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum_id)
                    if key_id in self.dht.storage:
                        del self.dht.storage[key_id]
                    
                    # Update active quorums list
                    self._update_active_list(quorum_id, add=False)
                    
                except Exception as e:
                    logger.warning(f"Failed to remove quorum from DHT: {e}")
            
            logger.info(f"Unregistered quorum {quorum_id[:8]}...")
            return True
    
    def _update_active_list(self, quorum_id: str, add: bool) -> None:
        """Update the active quorums list in DHT."""
        if not self.dht:
            return
        
        try:
            key_id = int(hashlib.sha1(self.DHT_KEY_QUORUM_LIST.encode()).hexdigest(), 16)
            
            # Get current list
            existing = self.dht.storage.get(key_id, "[]")
            active_ids = json.loads(existing)
            
            if add:
                if quorum_id not in active_ids:
                    active_ids.append(quorum_id)
            else:
                if quorum_id in active_ids:
                    active_ids.remove(quorum_id)
            
            self.dht.storage[key_id] = json.dumps(active_ids)
            
        except Exception as e:
            logger.debug(f"Failed to update active quorums list: {e}")
    
    def get_quorum(self, quorum_id: str) -> Optional[Quorum]:
        """
        Get a quorum by ID.
        
        Args:
            quorum_id: Quorum ID to look up
            
        Returns:
            Quorum if found, None otherwise
        """
        with self._lock:
            # Check local cache first
            if quorum_id in self.local_quorums:
                return self.local_quorums[quorum_id]
            
            # Try DHT lookup
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum_id)
                    
                    # Check DHT storage
                    if key_id in self.dht.storage:
                        json_str = self.dht.storage[key_id]
                        return Quorum.from_json(json_str)
                    
                    # Try network lookup
                    value = self.dht.lookup_value(key_id)
                    if value:
                        return Quorum.from_json(value)
                        
                except Exception as e:
                    logger.debug(f"Failed to lookup quorum from DHT: {e}")
            
            return None
    
    def get_active_quorums(self) -> List[Quorum]:
        """
        Get all active quorums.
        
        Returns:
            List of active Quorum objects
        """
        quorums = []
        
        with self._lock:
            # Get from local cache
            for quorum in self.local_quorums.values():
                if quorum.lifecycle == QuorumLifecycle.ACTIVE:
                    quorums.append(quorum)
            
            # Try to get more from DHT
            if self.dht:
                try:
                    key_id = int(hashlib.sha1(self.DHT_KEY_QUORUM_LIST.encode()).hexdigest(), 16)
                    existing = self.dht.storage.get(key_id, "[]")
                    active_ids = json.loads(existing)
                    
                    for qid in active_ids:
                        if qid not in self.local_quorums:
                            quorum = self.get_quorum(qid)
                            if quorum and quorum.lifecycle == QuorumLifecycle.ACTIVE:
                                quorums.append(quorum)
                                
                except Exception as e:
                    logger.debug(f"Failed to get active quorums from DHT: {e}")
        
        return quorums
    
    def discover_quorums(self) -> List[Quorum]:
        """
        Discover all active quorums in the network.
        
        Alias for get_active_quorums() for API compatibility.
        
        Returns:
            List of active quorums
        """
        return self.get_active_quorums()
    
    def find_quorums_by_speed_tier(self, speed_tier: str) -> List[Quorum]:
        """
        Find quorums matching a speed tier.
        
        Args:
            speed_tier: Speed tier to match (e.g., "tier2")
            
        Returns:
            List of matching quorums
        """
        all_quorums = self.get_active_quorums()
        return [q for q in all_quorums if q.speed_tier == speed_tier]
    
    def find_quorums_with_capacity(self) -> List[Quorum]:
        """
        Find quorums that have capacity for new members.
        
        Returns:
            List of quorums with available slots
        """
        all_quorums = self.get_active_quorums()
        return [q for q in all_quorums if len(q.members) < MAX_QUORUM_MEMBERS]
    
    def update_quorum(self, quorum: Quorum) -> bool:
        """
        Update a quorum in the registry.
        
        Args:
            quorum: Updated quorum
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            self.local_quorums[quorum.quorum_id] = quorum
            
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum.quorum_id)
                    self.dht.storage[key_id] = quorum.to_json()
                except Exception as e:
                    logger.warning(f"Failed to update quorum in DHT: {e}")
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            active = [q for q in self.local_quorums.values() if q.lifecycle == QuorumLifecycle.ACTIVE]
            return {
                "total_quorums": len(self.local_quorums),
                "active_quorums": len(active),
                "total_members": sum(len(q.members) for q in self.local_quorums.values()),
                "quorums_by_tier": {
                    tier: len([q for q in active if q.speed_tier == tier])
                    for tier in set(q.speed_tier for q in active)
                } if active else {},
            }


# =============================================================================
# QUORUM FORMATION SERVICE
# =============================================================================

class QuorumFormationService:
    """
    Service for forming and joining quorums.
    
    Implements the quorum formation algorithm:
    1. Get compatible speed tiers for initiator
    2. Query DHT for nodes in compatible tiers
    3. Find peers covering all layers
    4. Negotiate membership (propose + accept)
    5. Register quorum in DHT
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        layer_pool=None,
        dht_protocol=None,
    ):
        """
        Initialize the formation service.
        
        Args:
            registry: QuorumRegistry for storing formed quorums
            layer_pool: DynamicLayerPool for layer information
            dht_protocol: DHTProtocol for peer discovery
        """
        self.registry = registry
        self.layer_pool = layer_pool
        self.dht = dht_protocol
        self._lock = threading.RLock()
        
        # Formation state
        self._pending_proposals: Dict[str, Dict] = {}  # quorum_id -> proposal data
        self._proposal_responses: Dict[str, Dict[str, bool]] = {}  # quorum_id -> {node_id: accepted}
    
    def form_quorum(
        self,
        initiator_node_id: str,
        initiator_endpoint: str,
        initiator_layers: Tuple[int, int],
        initiator_speed_tier: str,
        total_layers: int,
        timeout: float = FORMATION_TIMEOUT,
    ) -> Optional[Quorum]:
        """
        Attempt to form a new quorum with the initiator as the first member.
        
        This is the main entry point for quorum formation:
        1. Check for existing compatible quorums to join
        2. If none, create a new quorum
        3. Find peers with compatible speed and covering remaining layers
        4. Propose quorum to potential members
        5. Wait for acceptance
        6. Register if enough members accept
        
        Args:
            initiator_node_id: Node ID of the initiating node
            initiator_endpoint: gRPC endpoint of initiator
            initiator_layers: (start, end) layer range of initiator
            initiator_speed_tier: Speed tier of initiator (e.g., "tier2")
            total_layers: Total number of layers in the model
            timeout: Timeout for formation in seconds
            
        Returns:
            Formed Quorum if successful, None if formation failed
        """
        logger.info(f"Attempting quorum formation: initiator={initiator_node_id[:8]}..., "
                   f"layers={initiator_layers}, tier={initiator_speed_tier}")
        
        # Step 1: Try to join an existing compatible quorum
        existing = self._find_joinable_quorum(
            initiator_speed_tier,
            initiator_layers,
        )
        if existing:
            # Join existing quorum
            member = QuorumMember(
                node_id=initiator_node_id,
                endpoint=initiator_endpoint,
                layer_range=initiator_layers,
                speed_tier=initiator_speed_tier,
            )
            if existing.add_member(member):
                self.registry.update_quorum(existing)
                logger.info(f"Joined existing quorum {existing.quorum_id[:8]}...")
                return existing
        
        # Step 2: Create new quorum
        quorum = Quorum(
            quorum_id=Quorum.generate_id(),
            speed_tier=initiator_speed_tier,
            arch_version=1,  # Will be updated from network state
            vocab_version=1,
        )
        
        # Add initiator as first member
        initiator_member = QuorumMember(
            node_id=initiator_node_id,
            endpoint=initiator_endpoint,
            layer_range=initiator_layers,
            speed_tier=initiator_speed_tier,
        )
        quorum.add_member(initiator_member)
        
        # Step 3: Find peers for remaining layers
        missing_layers = self._get_missing_layers(quorum, total_layers)
        
        if not missing_layers:
            # Initiator covers all layers - solo quorum
            quorum.lifecycle = QuorumLifecycle.ACTIVE
            self.registry.register_quorum(quorum)
            logger.info(f"Formed solo quorum {quorum.quorum_id[:8]}... (covers all {total_layers} layers)")
            return quorum
        
        # Step 4: Find compatible peers for missing layers
        compatible_peers = self._find_compatible_peers(
            initiator_speed_tier,
            missing_layers,
        )
        
        if not compatible_peers:
            # No compatible peers found - register anyway for later joining
            quorum.lifecycle = QuorumLifecycle.FORMING
            self.registry.register_quorum(quorum)
            logger.info(f"Registered forming quorum {quorum.quorum_id[:8]}... (waiting for peers)")
            return quorum
        
        # Step 5: Add compatible peers to quorum
        for peer in compatible_peers:
            if quorum.is_complete:
                break
            
            member = QuorumMember(
                node_id=peer["node_id"],
                endpoint=peer["endpoint"],
                layer_range=tuple(peer["layer_range"]),
                speed_tier=peer["speed_tier"],
            )
            quorum.add_member(member)
        
        # Step 6: Check if quorum is now complete
        if quorum.is_complete:
            quorum.lifecycle = QuorumLifecycle.ACTIVE
            logger.info(f"Formed complete quorum {quorum.quorum_id[:8]}... "
                       f"with {len(quorum.members)} members")
        else:
            quorum.lifecycle = QuorumLifecycle.FORMING
            logger.info(f"Formed incomplete quorum {quorum.quorum_id[:8]}... "
                       f"(missing layers: {self._get_missing_layers(quorum, total_layers)})")
        
        self.registry.register_quorum(quorum)
        return quorum
    
    def _find_joinable_quorum(
        self,
        speed_tier: str,
        layer_range: Tuple[int, int],
    ) -> Optional[Quorum]:
        """
        Find an existing quorum that the node can join.
        
        Criteria:
        - Same or compatible speed tier
        - Needs the layers we have
        - Has capacity for new members
        """
        # Get compatible quorums
        quorums = self.registry.find_quorums_by_speed_tier(speed_tier)
        
        for quorum in quorums:
            # Check capacity
            if len(quorum.members) >= MAX_QUORUM_MEMBERS:
                continue
            
            # Check if our layers would help
            current_coverage = quorum.layer_coverage
            our_layers = set(range(layer_range[0], layer_range[1]))
            
            # Would we add new layers?
            new_layers = our_layers - current_coverage
            if new_layers or not quorum.is_complete:
                return quorum
        
        return None
    
    def _get_missing_layers(self, quorum: Quorum, total_layers: int) -> Set[int]:
        """Get layers not covered by any quorum member."""
        all_layers = set(range(total_layers))
        covered = quorum.layer_coverage
        return all_layers - covered
    
    def _find_compatible_peers(
        self,
        speed_tier: str,
        missing_layers: Set[int],
    ) -> List[Dict[str, Any]]:
        """
        Find peers that can provide missing layers with compatible speed.
        
        Uses DHT to discover peers by layer.
        """
        from neuroshard.core.model.dynamic import COMPATIBLE_TIERS, SpeedTier
        
        compatible_peers = []
        seen_nodes = set()
        
        # Get compatible speed tiers
        try:
            tier_enum = SpeedTier(speed_tier)
            compatible_tiers = COMPATIBLE_TIERS.get(tier_enum, [tier_enum])
            compatible_tier_values = [t.value for t in compatible_tiers]
        except (ValueError, KeyError):
            compatible_tier_values = [speed_tier]
        
        if not self.dht:
            return []
        
        # Query DHT for each missing layer
        for layer_id in missing_layers:
            try:
                key_str = f"layer_{layer_id}"
                key_id = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
                
                # Check local DHT storage first
                value = self.dht.storage.get(key_id)
                if not value:
                    value = self.dht.lookup_value(key_id)
                
                if not value:
                    continue
                
                # Parse value (list of endpoints)
                try:
                    endpoints = json.loads(value)
                    if not isinstance(endpoints, list):
                        endpoints = [endpoints]
                except json.JSONDecodeError:
                    endpoints = [value]
                
                for endpoint in endpoints:
                    if endpoint in seen_nodes:
                        continue
                    seen_nodes.add(endpoint)
                    
                    # TODO: Query peer for their full info (speed tier, layer range)
                    # For now, assume compatible tier
                    peer_info = {
                        "node_id": hashlib.sha256(endpoint.encode()).hexdigest()[:16],
                        "endpoint": endpoint,
                        "layer_range": (layer_id, layer_id + 1),  # Single layer
                        "speed_tier": speed_tier,  # Assume compatible
                    }
                    compatible_peers.append(peer_info)
                    
            except Exception as e:
                logger.debug(f"Failed to query DHT for layer {layer_id}: {e}")
        
        return compatible_peers
    
    def try_join_quorum(
        self,
        node_id: str,
        endpoint: str,
        layer_range: Tuple[int, int],
        speed_tier: str,
        quorum_id: Optional[str] = None,
    ) -> Optional[Quorum]:
        """
        Try to join an existing quorum.
        
        Args:
            node_id: Node ID
            endpoint: gRPC endpoint
            layer_range: Layers this node holds
            speed_tier: Node's speed tier
            quorum_id: Specific quorum to join (optional)
            
        Returns:
            Joined Quorum if successful, None otherwise
        """
        if quorum_id:
            # Try to join specific quorum
            quorum = self.registry.get_quorum(quorum_id)
            if not quorum:
                return None
        else:
            # Find a compatible quorum
            quorum = self._find_joinable_quorum(speed_tier, layer_range)
            if not quorum:
                return None
        
        # Create member and add
        member = QuorumMember(
            node_id=node_id,
            endpoint=endpoint,
            layer_range=layer_range,
            speed_tier=speed_tier,
        )
        
        if quorum.add_member(member):
            # Check if quorum is now complete
            if quorum.is_complete and quorum.lifecycle == QuorumLifecycle.FORMING:
                quorum.lifecycle = QuorumLifecycle.ACTIVE
                logger.info(f"Quorum {quorum.quorum_id[:8]}... is now complete and active")
            
            self.registry.update_quorum(quorum)
            return quorum
        
        return None
    
    def leave_quorum(self, node_id: str, quorum_id: str) -> bool:
        """
        Leave a quorum gracefully.
        
        Args:
            node_id: Node ID leaving
            quorum_id: Quorum to leave
            
        Returns:
            True if left successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        member = quorum.remove_member(node_id)
        if not member:
            return False
        
        # Check if quorum should dissolve
        if len(quorum.members) == 0:
            quorum.lifecycle = QuorumLifecycle.DISSOLVED
            self.registry.unregister_quorum(quorum_id)
            logger.info(f"Quorum {quorum_id[:8]}... dissolved (no members)")
        elif not quorum.is_complete:
            # Quorum no longer complete
            quorum.lifecycle = QuorumLifecycle.FORMING
            self.registry.update_quorum(quorum)
            logger.info(f"Quorum {quorum_id[:8]}... no longer complete, seeking members")
        else:
            self.registry.update_quorum(quorum)
        
        return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_adaptive_min_quorum_size(network_size: int) -> int:
    """
    Get minimum quorum size based on network size.
    
    Adapts to network conditions:
    - Small network (< 5 nodes): 1 member minimum (allow solo)
    - Medium network (< 20 nodes): 2 members minimum
    - Large network (>= 20 nodes): 3 members minimum
    
    Args:
        network_size: Total number of nodes in network
        
    Returns:
        Minimum quorum size
    """
    if network_size < 5:
        return 1
    elif network_size < 20:
        return 2
    else:
        return 3


# =============================================================================
# QUORUM LIFECYCLE MANAGER
# =============================================================================

class QuorumLifecycleManager:
    """
    Manages the lifecycle of quorums.
    
    Responsibilities:
    - Monitor quorum health via heartbeats
    - Handle session renewal
    - Trigger dissolution when needed
    - Replace failed members
    
    Lifecycle States:
    FORMING -> ACTIVE -> RENEWING -> ACTIVE (if renewed)
                     -> DISSOLVING -> DISSOLVED
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        formation_service: QuorumFormationService,
    ):
        """
        Initialize the lifecycle manager.
        
        Args:
            registry: QuorumRegistry for quorum access
            formation_service: QuorumFormationService for member replacement
        """
        self.registry = registry
        self.formation = formation_service
        
        # Background task control
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 10.0  # Check every 10 seconds
    
    def start(self):
        """Start the lifecycle manager background tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="QuorumLifecycleMonitor"
        )
        self._monitor_thread.start()
        logger.info("Quorum lifecycle manager started")
    
    def stop(self):
        """Stop the lifecycle manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Quorum lifecycle manager stopped")
    
    def _monitor_loop(self):
        """Background loop to monitor quorum health."""
        while self._running:
            try:
                self._check_all_quorums()
            except Exception as e:
                logger.error(f"Error in quorum monitor: {e}")
            
            time.sleep(self._monitor_interval)
    
    def _check_all_quorums(self):
        """Check health and lifecycle of all quorums."""
        quorums = self.registry.get_active_quorums()
        
        for quorum in quorums:
            try:
                self._check_quorum_health(quorum)
                self._check_quorum_session(quorum)
            except Exception as e:
                logger.warning(f"Error checking quorum {quorum.quorum_id[:8]}...: {e}")
    
    def _check_quorum_health(self, quorum: Quorum):
        """Check health of a quorum and handle issues."""
        # Get stale and offline members
        stale = quorum.get_stale_members()
        offline = quorum.get_offline_members()
        
        # Handle offline members
        for member in offline:
            logger.warning(f"Member {member.node_id[:8]}... is offline in quorum {quorum.quorum_id[:8]}...")
            self._handle_member_failure(quorum, member)
        
        # Log stale members (warning only)
        for member in stale:
            if member not in offline:
                logger.debug(f"Member {member.node_id[:8]}... is stale in quorum {quorum.quorum_id[:8]}...")
    
    def _check_quorum_session(self, quorum: Quorum):
        """Check session timing and handle renewal/dissolution."""
        # Check if session expired
        if quorum.is_session_expired:
            self._handle_session_expired(quorum)
            return
        
        # Check if should check renewal
        if quorum.should_check_renewal and quorum.lifecycle == QuorumLifecycle.ACTIVE:
            self._handle_renewal_check(quorum)
    
    def _handle_member_failure(self, quorum: Quorum, member: QuorumMember):
        """Handle a failed member - try to replace or dissolve."""
        # Remove failed member
        quorum.remove_member(member.node_id)
        
        if not quorum.members:
            # No members left - dissolve
            self._dissolve_quorum(quorum, reason="all members failed")
            return
        
        if not quorum.is_complete:
            # Try to find replacement
            quorum.lifecycle = QuorumLifecycle.RENEWING
            
            # TODO: Trigger replacement search
            # For now, just mark as forming
            quorum.lifecycle = QuorumLifecycle.FORMING
            self.registry.update_quorum(quorum)
            logger.info(f"Quorum {quorum.quorum_id[:8]}... seeking replacement for failed member")
    
    def _handle_renewal_check(self, quorum: Quorum):
        """Handle session renewal check."""
        quorum.lifecycle = QuorumLifecycle.RENEWING
        
        # Check renewal conditions
        should_renew = self._should_renew_quorum(quorum)
        
        if should_renew:
            # Extend session
            quorum.extend_session()
            logger.info(f"Quorum {quorum.quorum_id[:8]}... session renewed")
        else:
            # Prepare for dissolution
            self._dissolve_quorum(quorum, reason="renewal conditions not met")
    
    def _should_renew_quorum(self, quorum: Quorum) -> bool:
        """
        Check if quorum should be renewed.
        
        Conditions for renewal:
        - All members healthy (not stale)
        - Processed minimum batches
        - Quorum still complete
        """
        # Check all members healthy
        if len(quorum.get_stale_members()) > 0:
            return False
        
        # Check minimum work done
        if quorum.total_batches < MIN_BATCHES_TO_RENEW:
            return False
        
        # Check still complete
        if not quorum.is_complete:
            return False
        
        return True
    
    def _handle_session_expired(self, quorum: Quorum):
        """Handle expired session - attempt renewal or dissolve."""
        if self._should_renew_quorum(quorum):
            quorum.extend_session()
            logger.info(f"Quorum {quorum.quorum_id[:8]}... session auto-renewed on expiry")
        else:
            self._dissolve_quorum(quorum, reason="session expired")
    
    def _dissolve_quorum(self, quorum: Quorum, reason: str = ""):
        """Dissolve a quorum gracefully."""
        logger.info(f"Dissolving quorum {quorum.quorum_id[:8]}...: {reason}")
        
        quorum.lifecycle = QuorumLifecycle.DISSOLVING
        self.registry.update_quorum(quorum)
        
        # TODO: Notify members to save weights, find new quorums
        
        # Mark as dissolved
        quorum.lifecycle = QuorumLifecycle.DISSOLVED
        self.registry.unregister_quorum(quorum.quorum_id)
    
    def process_heartbeat(self, quorum_id: str, node_id: str) -> bool:
        """
        Process a heartbeat from a quorum member.
        
        Args:
            quorum_id: Quorum ID
            node_id: Node sending heartbeat
            
        Returns:
            True if processed successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        success = quorum.update_heartbeat(node_id)
        if success:
            self.registry.update_quorum(quorum)
        
        return success
    
    def record_batch(self, quorum_id: str, batches: int = 1) -> bool:
        """
        Record processed batches for a quorum.
        
        Args:
            quorum_id: Quorum ID
            batches: Number of batches processed
            
        Returns:
            True if recorded successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        quorum.record_batch(batches)
        self.registry.update_quorum(quorum)
        return True
    
    def get_quorum_status(self, quorum_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a quorum.
        
        Args:
            quorum_id: Quorum ID
            
        Returns:
            Status dict or None if quorum not found
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return None
        
        return {
            "quorum_id": quorum.quorum_id,
            "lifecycle": quorum.lifecycle.value,
            "speed_tier": quorum.speed_tier,
            "member_count": len(quorum.members),
            "healthy_members": len(quorum.get_healthy_members()),
            "stale_members": len(quorum.get_stale_members()),
            "is_complete": quorum.is_complete,
            "total_batches": quorum.total_batches,
            "batches_per_second": quorum.batches_per_second,
            "session_remaining": quorum.session_remaining,
            "session_progress": quorum.session_progress,
        }


# =============================================================================
# QUORUM TRAINER
# =============================================================================

# DiLoCo sync settings
SYNC_INTERVAL = 500              # Batches between cohort syncs
OUTER_LR = 0.7                   # Outer learning rate for DiLoCo


class QuorumTrainer:
    """
    Training coordinator for quorum-based training.
    
    Responsibilities:
    - Manage within-quorum synchronous pipeline training
    - Coordinate DiLoCo cohort sync across quorums
    - Track training progress and generate proofs
    
    Pipeline Flow (within quorum):
    1. INITIATOR: Get batch, embed, forward through layers, send to next
    2. PROCESSOR: Receive activations, forward, send to next
    3. FINISHER: Compute loss, backward, send gradients back
    4. All: Apply gradients locally
    
    DiLoCo Sync (across quorums):
    - Every SYNC_INTERVAL batches
    - Compute pseudo-gradient (initial - current weights)
    - Exchange with cohort (other nodes holding same layers)
    - Weighted aggregation (sqrt(n) * freshness)
    - Apply outer update
    """
    
    def __init__(
        self,
        quorum: Quorum,
        node_id: str,
        model: Any,
        optimizer: Any,
        genesis_loader: Any = None,
        dht_protocol: Any = None,
    ):
        """
        Initialize QuorumTrainer.
        
        Args:
            quorum: The quorum this trainer belongs to
            node_id: This node's ID
            model: The model to train (DynamicNeuroLLM)
            optimizer: PyTorch optimizer
            genesis_loader: Genesis data loader (for initiator)
            dht_protocol: DHT protocol for cohort discovery
        """
        self.quorum = quorum
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.genesis_loader = genesis_loader
        self.dht = dht_protocol
        
        # Determine role
        self.member = quorum.get_member(node_id)
        if not self.member:
            raise ValueError(f"Node {node_id} is not a member of quorum {quorum.quorum_id}")
        
        self.is_initiator = self.member.role == QuorumRole.INITIATOR
        self.is_finisher = self.member.role == QuorumRole.FINISHER
        
        # DiLoCo state
        self.initial_weights: Dict[str, Any] = {}
        self.batches_since_sync = 0
        self.sync_round = 0
        
        # Training state
        self.running = False
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Stats
        self.total_batches = 0
        self.current_loss: Optional[float] = None
        
        # Snapshot initial weights for DiLoCo
        self._snapshot_weights()
        
        logger.info(f"QuorumTrainer initialized: quorum={quorum.quorum_id[:8]}..., "
                   f"role={self.member.role.value}, layers={self.member.layer_range}")
    
    def _snapshot_weights(self):
        """Snapshot current weights for DiLoCo pseudo-gradient computation."""
        import torch
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    def start(self):
        """Start the training loop."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name=f"QuorumTrainer-{self.quorum.quorum_id[:8]}"
        )
        self._training_thread.start()
        logger.info("QuorumTrainer started")
    
    def stop(self):
        """Stop the training loop gracefully."""
        self.running = False
        self._stop_event.set()
        if self._training_thread:
            self._training_thread.join(timeout=5.0)
        logger.info("QuorumTrainer stopped")
    
    def _training_loop(self):
        """Main training loop for quorum member."""
        while self.running and self.quorum.lifecycle == QuorumLifecycle.ACTIVE:
            try:
                if self.is_initiator:
                    self._initiate_batch()
                else:
                    # Wait for activation from previous node
                    self._process_received_activation()
                
                self.batches_since_sync += 1
                self.total_batches += 1
                self.quorum.record_batch()
                
                # Check if time for DiLoCo sync
                if self.batches_since_sync >= SYNC_INTERVAL:
                    self._cohort_sync()
                
                # Update member stats
                if self.member:
                    self.member.batches_processed = self.total_batches
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(1)  # Brief pause on error
            
            # Check for stop signal
            if self._stop_event.is_set():
                break
        
        logger.info(f"Training loop ended: {self.total_batches} batches processed")
    
    def _initiate_batch(self):
        """Initiator: Get batch, embed, and start pipeline."""
        import torch
        
        if not self.genesis_loader:
            logger.warning("No genesis loader - cannot initiate batch")
            time.sleep(1)
            return
        
        # Get batch from genesis loader
        batch = self.genesis_loader.get_batch()
        if batch is None:
            time.sleep(0.1)
            return
        
        # Forward through local layers (embedding + initial layers)
        with torch.no_grad() if not self.model.training else torch.enable_grad():
            hidden = self.model.forward_pipeline(batch['input_ids'])
        
        # Send to next node in pipeline
        next_hop = self.quorum.get_next_hop(self.node_id)
        if next_hop:
            self._send_activation(next_hop, hidden, batch)
        elif self.is_finisher:
            # Solo quorum - do full forward+backward
            self._finisher_backward(hidden, batch)
    
    def _process_received_activation(self):
        """Processor/Finisher: Process received activation."""
        # This would integrate with the SwarmServiceMixin to receive activations
        # For now, this is a placeholder that shows the structure
        time.sleep(0.1)  # Wait for activation
    
    def _send_activation(self, target: QuorumMember, hidden: Any, batch: Dict):
        """Send activation to next node in pipeline."""
        # This would use gRPC to send activations
        # Integrates with SwarmForward RPC
        pass
    
    def _finisher_backward(self, hidden: Any, batch: Dict):
        """Finisher: Compute loss and start backward pass."""
        import torch
        
        # Compute loss
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(hidden, batch.get('labels'))
        else:
            # Placeholder
            loss = torch.tensor(0.0)
        
        self.current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
        
        # Backward pass
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def _cohort_sync(self):
        """Synchronize with cohort (DiLoCo cross-quorum sync)."""
        import torch
        
        logger.info(f"Starting cohort sync round {self.sync_round}")
        
        # 1. Compute pseudo-gradient
        pseudo_grad = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                pseudo_grad[name] = self.initial_weights[name] - param.data
        
        # 2. Find cohort (other nodes with same layers, different quorums)
        cohort = self._find_layer_cohort()
        
        # 3. Exchange and aggregate (simplified - would use actual network)
        # In full implementation, this exchanges with cohort via gRPC
        aggregated = pseudo_grad  # For now, just use local
        
        # 4. Apply outer update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_weights and name in aggregated:
                    param.data = self.initial_weights[name] + OUTER_LR * aggregated[name]
        
        # 5. Reset for next round
        self._snapshot_weights()
        self.batches_since_sync = 0
        self.sync_round += 1
        
        logger.info(f"Cohort sync complete: round {self.sync_round}")
    
    def _find_layer_cohort(self) -> List[str]:
        """Find nodes holding the same layers in other quorums."""
        if not self.dht:
            return []
        
        cohort = []
        layer_start, layer_end = self.member.layer_range
        
        # Query DHT for each layer
        for layer_id in range(layer_start, layer_end):
            try:
                key_str = f"layer_{layer_id}"
                key_id = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
                
                if key_id in self.dht.storage:
                    import json
                    value = self.dht.storage[key_id]
                    endpoints = json.loads(value) if isinstance(value, str) else [value]
                    
                    for endpoint in endpoints:
                        if endpoint not in cohort:
                            cohort.append(endpoint)
            except Exception:
                pass
        
        # Remove self
        my_endpoint = self.member.endpoint
        return [c for c in cohort if c != my_endpoint]
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "quorum_id": self.quorum.quorum_id,
            "node_id": self.node_id,
            "role": self.member.role.value if self.member else "unknown",
            "total_batches": self.total_batches,
            "batches_since_sync": self.batches_since_sync,
            "sync_round": self.sync_round,
            "current_loss": self.current_loss,
            "running": self.running,
        }


# =============================================================================
# QUORUM INFERENCE ROUTER
# =============================================================================

# Inference pricing constants
BASE_INFERENCE_PRICE = 0.0001      # NEURO per token
SPEED_MULTIPLIERS = {
    "tier1": 1.5,
    "tier2": 1.3,
    "tier3": 1.0,
    "tier4": 0.8,
    "tier5": 0.6,
}


@dataclass
class InferenceQuorumInfo:
    """Information about a quorum available for inference."""
    quorum_id: str
    speed_tier: str
    initiator_endpoint: str
    estimated_latency_ms: float
    price_per_token: float
    available_capacity: int
    reputation: float = 0.95
    
    def to_json(self) -> str:
        return json.dumps({
            "quorum_id": self.quorum_id,
            "speed_tier": self.speed_tier,
            "initiator_endpoint": self.initiator_endpoint,
            "estimated_latency_ms": self.estimated_latency_ms,
            "price_per_token": self.price_per_token,
            "available_capacity": self.available_capacity,
            "reputation": self.reputation,
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'InferenceQuorumInfo':
        data = json.loads(json_str)
        return cls(**data)


def calculate_inference_price(speed_tier: str, utilization: float = 0.5, reputation: float = 0.95) -> float:
    """
    Calculate per-token price for inference.
    
    Args:
        speed_tier: Quorum speed tier (tier1-tier5)
        utilization: Current utilization (0-1)
        reputation: Quorum reputation score (0-1)
        
    Returns:
        Price per token in NEURO
    """
    # Speed multiplier (faster = more expensive)
    speed_mult = SPEED_MULTIPLIERS.get(speed_tier, 1.0)
    
    # Demand multiplier (higher utilization = higher price)
    demand_mult = 1.0 + utilization ** 2  # 1.0 to 2.0
    
    # Reputation multiplier (proven quorums charge more)
    rep_mult = 0.8 + 0.4 * reputation  # 0.8 to 1.2
    
    price = BASE_INFERENCE_PRICE * demand_mult * speed_mult * rep_mult
    
    # Ensure minimum profitability
    min_price = BASE_INFERENCE_PRICE * 0.5
    return max(price, min_price)


class QuorumInferenceRouter:
    """
    Routes inference requests to appropriate quorums.
    
    Discovery Flow:
    1. Query DHT for available inference quorums
    2. Filter by capacity and status (ACTIVE only)
    3. Score by latency, price, and reputation
    4. Return ranked list for client selection
    
    Execution Flow:
    1. Client sends request to selected quorum's initiator
    2. Initiator embeds and forwards through pipeline
    3. Finisher generates output, sends back
    4. Payment distributed to quorum members
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        dht_protocol: Any = None,
    ):
        """
        Initialize QuorumInferenceRouter.
        
        Args:
            registry: QuorumRegistry for quorum lookup
            dht_protocol: DHT for network-wide discovery
        """
        self.registry = registry
        self.dht = dht_protocol
        
        # Cache of available quorums
        self._quorum_cache: Dict[str, InferenceQuorumInfo] = {}
        self._cache_ttl = 30.0  # 30 second cache
        self._last_cache_update = 0.0
        
        logger.info("QuorumInferenceRouter initialized")
    
    def discover_quorums(self, force_refresh: bool = False) -> List[InferenceQuorumInfo]:
        """
        Discover available inference quorums.
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of available quorums sorted by score
        """
        now = time.time()
        
        # Check cache
        if not force_refresh and (now - self._last_cache_update) < self._cache_ttl:
            return list(self._quorum_cache.values())
        
        # Refresh from registry
        self._quorum_cache.clear()
        
        # Get all active quorums
        all_quorums = self.registry.discover_quorums()
        
        for quorum in all_quorums:
            if quorum.lifecycle != QuorumLifecycle.ACTIVE:
                continue
            
            if not quorum.is_complete:
                continue
            
            # Find initiator
            initiator = None
            for member in quorum.members:
                if member.role == QuorumRole.INITIATOR:
                    initiator = member
                    break
            
            if not initiator:
                continue
            
            # Calculate price
            utilization = 0.5  # TODO: Get actual utilization
            price = calculate_inference_price(quorum.speed_tier, utilization)
            
            # Estimate latency based on speed tier
            latency_estimates = {
                "tier1": 50,
                "tier2": 150,
                "tier3": 400,
                "tier4": 1500,
                "tier5": 5000,
            }
            latency = latency_estimates.get(quorum.speed_tier, 1000)
            
            info = InferenceQuorumInfo(
                quorum_id=quorum.quorum_id,
                speed_tier=quorum.speed_tier,
                initiator_endpoint=initiator.endpoint,
                estimated_latency_ms=latency,
                price_per_token=price,
                available_capacity=10,  # TODO: Get actual capacity
            )
            
            self._quorum_cache[quorum.quorum_id] = info
        
        self._last_cache_update = now
        
        # Sort by score (lower is better: latency + price penalty)
        result = list(self._quorum_cache.values())
        result.sort(key=lambda q: q.estimated_latency_ms + q.price_per_token * 10000)
        
        return result
    
    def select_best_quorum(
        self,
        max_latency_ms: Optional[float] = None,
        max_price: Optional[float] = None,
        min_reputation: float = 0.8,
    ) -> Optional[InferenceQuorumInfo]:
        """
        Select the best quorum for an inference request.
        
        Args:
            max_latency_ms: Maximum acceptable latency
            max_price: Maximum acceptable price per token
            min_reputation: Minimum reputation score
            
        Returns:
            Best matching quorum or None
        """
        quorums = self.discover_quorums()
        
        for quorum in quorums:
            # Apply filters
            if max_latency_ms and quorum.estimated_latency_ms > max_latency_ms:
                continue
            
            if max_price and quorum.price_per_token > max_price:
                continue
            
            if quorum.reputation < min_reputation:
                continue
            
            if quorum.available_capacity <= 0:
                continue
            
            return quorum
        
        return None
    
    def get_quorum_for_routing(self, quorum_id: str) -> Optional[Quorum]:
        """
        Get full quorum for routing an inference request.
        
        Args:
            quorum_id: Quorum to route to
            
        Returns:
            Quorum object or None
        """
        return self.registry.get_quorum(quorum_id)
    
    def estimate_cost(self, quorum_id: str, max_tokens: int) -> float:
        """
        Estimate inference cost for a request.
        
        Args:
            quorum_id: Target quorum
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated cost in NEURO
        """
        if quorum_id in self._quorum_cache:
            price = self._quorum_cache[quorum_id].price_per_token
            return price * max_tokens
        return BASE_INFERENCE_PRICE * max_tokens


# =============================================================================
# ASYNC TRAINER - For T5 nodes and nodes without quorum
# =============================================================================

# Freshness decay for async contributions (from whitepaper)
ASYNC_FRESHNESS_DECAY = {
    3600: 1.0,      # < 1 hour: full value
    86400: 0.7,     # < 1 day: 70% value
    604800: 0.5,    # < 1 week: 50% value
    float('inf'): 0.3,  # > 1 week: 30% value
}


def calculate_async_freshness(age_seconds: float) -> float:
    """Calculate freshness multiplier for async gradient contributions."""
    for threshold, freshness in sorted(ASYNC_FRESHNESS_DECAY.items()):
        if age_seconds < threshold:
            return freshness
    return 0.3


class AsyncTrainer:
    """
    Async training for nodes that can't join real-time quorums.
    
    Per whitepaper, async training is for:
    - T5 (slow) nodes that can't keep up with real-time pipelines
    - Any node that can't find a compatible quorum
    
    Process:
    1. Download current weights for held layers
    2. Train locally for N steps using Genesis data
    3. Compute pseudo-gradient: delta_w = w_initial - w_final
    4. Submit to layer cohort for next DiLoCo sync round
    5. Apply freshness decay to contribution weight
    
    Async contributions are weighted by: sqrt(batches) * freshness
    This ensures fair influence while prioritizing recent work.
    """
    
    # Training interval: how often to do async training
    ASYNC_TRAIN_INTERVAL = 60  # Train every 60 seconds
    ASYNC_BATCH_SIZE = 4       # Smaller batches for async training
    ASYNC_STEPS_PER_ROUND = 50 # Steps per async training round
    
    def __init__(
        self,
        node_id: str,
        model: Any,
        optimizer: Any,
        genesis_loader: Any = None,
        dht_protocol: Any = None,
    ):
        """
        Initialize AsyncTrainer.
        
        Args:
            node_id: This node's ID
            model: The model to train
            optimizer: PyTorch optimizer
            genesis_loader: Genesis data loader
            dht_protocol: DHT for cohort discovery
        """
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.genesis_loader = genesis_loader
        self.dht = dht_protocol
        
        # Training state
        self.running = False
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # DiLoCo-like state for async
        self.initial_weights: Dict[str, Any] = {}
        self.last_sync_time = time.time()
        
        # Stats
        self.total_batches = 0
        self.total_syncs = 0
        self.current_loss: Optional[float] = None
        self.last_gradient_submission_time: Optional[float] = None
        
        # Snapshot initial weights
        self._snapshot_weights()
        
        logger.info(f"AsyncTrainer initialized for node {node_id[:16]}...")
    
    def _snapshot_weights(self):
        """Snapshot current weights for pseudo-gradient computation."""
        import torch
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    def start(self):
        """Start the async training loop."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._async_training_loop,
            daemon=True,
            name=f"AsyncTrainer-{self.node_id[:8]}"
        )
        self._training_thread.start()
        logger.info("AsyncTrainer started")
    
    def stop(self):
        """Stop the async training loop."""
        self.running = False
        self._stop_event.set()
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=5.0)
        logger.info("AsyncTrainer stopped")
    
    def _async_training_loop(self):
        """
        Main async training loop.
        
        Unlike QuorumTrainer, this runs independently:
        1. Train locally for ASYNC_STEPS_PER_ROUND steps
        2. Compute pseudo-gradient
        3. Submit to DiLoCo cohort sync (if available)
        4. Reset and repeat
        """
        import torch
        
        logger.info("[ASYNC] Starting async training loop...")
        
        # Track if we've logged the "no genesis loader" message
        _logged_no_genesis = False
        
        while self.running and not self._stop_event.is_set():
            try:
                # Check if we have Genesis data to train on
                if not self.genesis_loader:
                    if not _logged_no_genesis:
                        logger.warning("[ASYNC] No Genesis data loader available - waiting for data...")
                        logger.info("[ASYNC] Genesis loader will be initialized when node downloads training shards")
                        _logged_no_genesis = True
                    time.sleep(self.ASYNC_TRAIN_INTERVAL)
                    continue
                
                _logged_no_genesis = False  # Reset if we get a loader
                
                # Train for N steps
                losses = []
                data_not_ready_count = 0
                
                for step in range(self.ASYNC_STEPS_PER_ROUND):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        # Get batch from Genesis loader
                        # Note: get_batch() raises RuntimeError when data not ready, doesn't return None
                        input_ids, labels = self.genesis_loader.get_batch(batch_size=self.ASYNC_BATCH_SIZE)
                        
                        # Move to device
                        device = next(self.model.parameters()).device
                        input_ids = input_ids.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass through local layers
                        self.model.train()
                        
                        # Use forward_my_layers for proper layer handling
                        embeddings = self.model.embed(input_ids)
                        hidden = self.model.forward_my_layers(embeddings)
                        logits = self.model.compute_logits(hidden)
                        
                        # Compute loss
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Optimizer step
                        self.optimizer.step()
                        
                        # Track stats
                        self.total_batches += 1
                        losses.append(loss.item())
                        
                        # Log every 10 steps
                        if self.total_batches % 10 == 0:
                            logger.info(f"[ASYNC] Step {self.total_batches}: loss={loss.item():.4f}")
                        
                    except RuntimeError as e:
                        if "Data not ready" in str(e) or "shard" in str(e).lower():
                            data_not_ready_count += 1
                            time.sleep(0.1)  # Brief wait for data
                        else:
                            logger.warning(f"[ASYNC] Training step error: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"[ASYNC] Training step error: {e}")
                        continue
                
                # Log data availability issues
                if data_not_ready_count > 0 and len(losses) == 0:
                    logger.info(f"[ASYNC] Data not ready ({data_not_ready_count} attempts) - waiting for shard download")
                
                # Update current loss
                if losses:
                    self.current_loss = sum(losses) / len(losses)
                    logger.info(f"[ASYNC] Completed {len(losses)} steps, avg_loss={self.current_loss:.4f}")
                
                # Compute pseudo-gradient
                pseudo_gradient = self._compute_pseudo_gradient()
                
                # Submit to cohort sync (if DHT available)
                if pseudo_gradient and self.dht:
                    self._submit_to_cohort_sync(pseudo_gradient)
                
                # Reset weights snapshot for next round
                self._snapshot_weights()
                
                # Wait before next training round
                time.sleep(self.ASYNC_TRAIN_INTERVAL)
                
            except Exception as e:
                logger.error(f"[ASYNC] Training loop error: {e}")
                time.sleep(10)
        
        logger.info("[ASYNC] Training loop ended")
    
    def _compute_pseudo_gradient(self) -> Optional[Dict[str, Any]]:
        """
        Compute pseudo-gradient: delta_w = w_initial - w_current
        
        This represents the accumulated update from local training.
        """
        import torch
        
        if not self.initial_weights:
            return None
        
        pseudo_grad = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_weights and param.requires_grad:
                # Pseudo-gradient is the difference (what we learned)
                pseudo_grad[name] = self.initial_weights[name] - param.data
        
        return pseudo_grad
    
    def _submit_to_cohort_sync(self, pseudo_gradient: Dict[str, Any]):
        """
        Submit pseudo-gradient to layer cohort for DiLoCo sync.
        
        Async contributions are weighted by freshness:
        - < 1 hour: 100%
        - < 1 day: 70%
        - < 1 week: 50%
        - > 1 week: 30%
        """
        try:
            # Calculate freshness
            age_seconds = time.time() - self.last_sync_time
            freshness = calculate_async_freshness(age_seconds)
            
            # Create gradient contribution
            contribution = {
                "node_id": self.node_id,
                "batches": self.total_batches,
                "timestamp": time.time(),
                "freshness": freshness,
                "is_async": True,
                # Note: Actual gradient data would be compressed and sent via DHT/P2P
            }
            
            # Log submission
            logger.info(f"[ASYNC] Submitting gradient: batches={self.total_batches}, "
                       f"freshness={freshness:.2f}, age={age_seconds/60:.1f}min")
            
            # Update tracking
            self.last_gradient_submission_time = time.time()
            self.last_sync_time = time.time()
            self.total_syncs += 1
            
            # TODO: Actual submission via DHT/P2P to layer cohort
            # For now, the gradient is computed and ready for submission
            
        except Exception as e:
            logger.error(f"[ASYNC] Failed to submit gradient: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async trainer statistics."""
        return {
            "running": self.running,
            "total_batches": self.total_batches,
            "total_syncs": self.total_syncs,
            "current_loss": self.current_loss,
            "last_submission": self.last_gradient_submission_time,
            "is_async": True,
        }
