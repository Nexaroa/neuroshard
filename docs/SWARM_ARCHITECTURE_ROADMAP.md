# NeuroShard Swarm Architecture Roadmap

## Strategic Pivot: Throughput over Latency

**Core Philosophy**: "Compute is cheap; Bandwidth is expensive."

**Goal**: 95% GPU utilization, even if model converges 10% slower in wall-clock time per epoch. We win by being unstoppable, not by being the fastest.

---

## Executive Summary

This document outlines the architectural pivot from NeuroShard's current **synchronous pipeline** approach to an **asynchronous, swarm-based** architecture. The changes are organized into three technical epics:

| Epic | Objective | Priority |
|------|-----------|----------|
| **Epic 1: Swarm Network Layer** | Fault-tolerant multipath routing | HIGH |
| **Epic 2: Async Micro-Batch Engine** | Decouple compute from communication | HIGH |
| **Epic 3: Lazy Gradient Sync** | 90%+ bandwidth reduction via DiLoCo-style accumulation | MEDIUM |

---

## Current Architecture Analysis

### What We Have

```
┌─────────────────────────────────────────────────────────────────┐
│ CURRENT: Synchronous Pipeline (Brittle)                         │
│                                                                 │
│   [Driver]  ──→  [Worker 1]  ──→  [Worker 2]  ──→  [Validator] │
│   Layer 0        Layers 1-N       Layers N-M       Last Layer   │
│                                                                 │
│   PROBLEM: If Worker 1 hangs, entire pipeline crashes           │
└─────────────────────────────────────────────────────────────────┘
```

**Files Involved:**
- `neuroshard/core/dynamic_model.py` - DynamicLayerPool, pipeline routing
- `neuroshard/core/p2p.py` - get_next_hop(), single-peer routing
- `neuroshard/core/distributed_training.py` - TrainingCoordinator, gradient gossip
- `neuroshard/core/gradient_gossip.py` - GossipProtocol
- `runner.py` - Training loop (blocking train_step)

### Pain Points

1. **Single Point of Failure**: `get_next_hop()` returns ONE peer. If it fails, inference/training crashes.
2. **Synchronous Training**: `train_step()` blocks until complete. GPU starvation during network I/O.
3. **Frequent Gradient Sync**: Every training step gossips gradients. Bandwidth-hungry on residential connections.
4. **No Capacity Awareness**: Peers don't advertise queue depth or available compute.

---

## Epic 1: Swarm Network Layer (Dynamic Routing)

### Objective
Replace brittle linear pipeline with fault-tolerant multipath routing.

**Key Directive**: "If Node A hangs, the packet must automatically flow to Node B without crashing the run."

### Architecture Target

```
┌─────────────────────────────────────────────────────────────────┐
│ TARGET: Swarm Routing (Fault-Tolerant)                          │
│                                                                 │
│   [Driver A]  ──┬──→  [Worker Pool 1]  ──┬──→  [Validator Pool] │
│   [Driver B]  ──┘     (K=3 replicas)   ──┘     (K=2 replicas)   │
│                                                                 │
│   BEHAVIOR: Automatic failover within 200ms                     │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Tasks

#### Task 1.1: Multipath Routing Logic
**File**: `neuroshard/core/swarm_router.py` (NEW)

```python
@dataclass
class PeerCandidate:
    """A candidate peer for routing."""
    node_id: str
    grpc_addr: str
    layer_range: Tuple[int, int]  # (start, end)
    latency_ms: float             # Rolling average
    queue_depth: int              # Current queue size
    last_heartbeat: float
    
    def score(self, weights: Dict[str, float] = None) -> float:
        """Weighted score: lower is better."""
        w = weights or {"latency": 0.4, "queue": 0.6}
        # Normalize: latency 0-500ms, queue 0-100
        latency_norm = min(1.0, self.latency_ms / 500)
        queue_norm = min(1.0, self.queue_depth / 100)
        return w["latency"] * latency_norm + w["queue"] * queue_norm


class SwarmRouter:
    """Fault-tolerant multipath router for distributed inference/training."""
    
    ACK_TIMEOUT_MS = 200       # Failover if no ACK in 200ms
    K_CANDIDATES = 3           # Return top-K candidates per layer range
    
    def __init__(self, dht_protocol, layer_pool):
        self.dht = dht_protocol
        self.layer_pool = layer_pool
        self.peer_stats: Dict[str, PeerCandidate] = {}
        
    def get_candidates(self, target_layer: int) -> List[PeerCandidate]:
        """
        Get K candidates for a target layer, sorted by score.
        
        Returns candidates from:
        1. Local cache (fastest)
        2. DHT lookup (if cache miss)
        3. Tracker fallback (if DHT fails)
        """
        candidates = []
        
        # Strategy 1: Local cache
        for node_id, info in self.peer_stats.items():
            if info.layer_range[0] <= target_layer < info.layer_range[1]:
                candidates.append(info)
        
        # Strategy 2: DHT lookup (if not enough candidates)
        if len(candidates) < self.K_CANDIDATES:
            dht_peers = self._dht_lookup_layer(target_layer)
            candidates.extend(dht_peers)
        
        # Sort by score, return top K
        candidates.sort(key=lambda c: c.score())
        return candidates[:self.K_CANDIDATES]
    
    async def send_with_failover(
        self, 
        tensor: torch.Tensor, 
        target_layer: int,
        session_id: str
    ) -> torch.Tensor:
        """
        Send tensor to target layer with automatic failover.
        
        1. Try primary candidate
        2. If no ACK in 200ms, try secondary
        3. Continue until success or all candidates exhausted
        """
        candidates = self.get_candidates(target_layer)
        
        if not candidates:
            raise RuntimeError(f"No candidates for layer {target_layer}")
        
        last_error = None
        for i, candidate in enumerate(candidates):
            try:
                result = await asyncio.wait_for(
                    self._send_to_peer(candidate.grpc_addr, tensor, session_id),
                    timeout=self.ACK_TIMEOUT_MS / 1000
                )
                
                # Update stats on success
                self._update_peer_stats(candidate.node_id, success=True)
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Peer {candidate.node_id[:8]}... timed out, trying next")
                self._update_peer_stats(candidate.node_id, success=False)
                last_error = TimeoutError(f"Peer {candidate.node_id[:8]} timed out")
                continue
            except Exception as e:
                logger.warning(f"Peer {candidate.node_id[:8]}... failed: {e}")
                self._update_peer_stats(candidate.node_id, success=False)
                last_error = e
                continue
        
        raise RuntimeError(f"All {len(candidates)} candidates failed for layer {target_layer}") from last_error
```

#### Task 1.2: Capacity Bitmask Heartbeats
**File**: `neuroshard/core/swarm_heartbeat.py` (NEW)

```python
@dataclass
class CapacityBitmask:
    """Lightweight capacity advertisement broadcast every 5 seconds."""
    node_id: str
    timestamp: float
    
    # Capacity info (fits in single UDP packet)
    available_memory_mb: int      # 4 bytes
    queue_depth: int              # 2 bytes (0-65535)
    layer_range: Tuple[int, int]  # 4 bytes (2 x uint16)
    gpu_utilization: float        # 1 byte (0-100%)
    network_saturation: float     # 1 byte (0-100%)
    
    # Status flags
    is_training: bool
    is_accepting_inference: bool
    is_accepting_activations: bool
    
    def to_bytes(self) -> bytes:
        """Serialize to ~20 bytes for UDP broadcast."""
        # ...compact binary format...
        
    @classmethod 
    def from_bytes(cls, data: bytes) -> 'CapacityBitmask':
        """Deserialize from UDP packet."""
        # ...


class SwarmHeartbeatService:
    """Broadcasts and receives capacity heartbeats."""
    
    HEARTBEAT_INTERVAL = 5.0  # seconds
    STALE_THRESHOLD = 15.0    # Consider peer dead after 15s
    
    def __init__(self, node_id: str, udp_port: int = 9999):
        self.node_id = node_id
        self.udp_port = udp_port
        self.peer_capacities: Dict[str, CapacityBitmask] = {}
        self.running = False
        
    def start(self):
        """Start heartbeat broadcast and listener."""
        self.running = True
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
    def get_available_peers(self, min_memory: int = 0) -> List[CapacityBitmask]:
        """Get peers with available capacity."""
        now = time.time()
        return [
            cap for cap in self.peer_capacities.values()
            if (now - cap.timestamp) < self.STALE_THRESHOLD
            and cap.available_memory_mb >= min_memory
            and cap.is_accepting_activations
        ]
```

#### Task 1.3: Update DHT Lookup
**File**: `neuroshard/core/dht_protocol.py` (MODIFY)

```python
# BEFORE:
def lookup_value(self, key: str) -> Optional[str]:
    """Returns single value."""
    
# AFTER:
def lookup_values(self, key: str, k: int = 3) -> List[str]:
    """Returns up to K values sorted by proximity/freshness."""
```

#### Task 1.4: Update Layer Pool
**File**: `neuroshard/core/dynamic_model.py` (MODIFY)

```python
# BEFORE (DynamicLayerPool.get_pipeline_route):
def get_pipeline_route(self) -> List[Tuple[int, str]]:
    """Returns single grpc_addr per layer."""
    
# AFTER:
def get_swarm_route(self) -> Dict[int, List[PeerCandidate]]:
    """Returns K candidates per layer for swarm routing."""
```

#### Task 1.5: NAT Traversal & Connectivity
**File**: `neuroshard/core/nat_traversal.py` (NEW)

**Context**: Most nodes will be behind residential NATs. Direct gRPC/UDP calls will fail without proper NAT traversal.

**Metric**: Connection success rate > 90% across different ISPs.

**Option A: libp2p Integration (Recommended)**
```python
"""
NAT Traversal using libp2p.

libp2p provides:
- AutoNAT: Detect NAT type and public reachability
- Circuit Relay: Relay traffic through public nodes when direct fails
- Hole Punching: Coordinate simultaneous open for direct P2P
- mDNS: Local network peer discovery

Dependencies: py-libp2p or libp2p via subprocess/FFI
"""
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Callable

class NATType(Enum):
    """Detected NAT type."""
    UNKNOWN = "unknown"
    OPEN = "open"                    # Direct connection works
    FULL_CONE = "full_cone"          # Easy hole punch
    RESTRICTED_CONE = "restricted"   # Medium difficulty
    PORT_RESTRICTED = "port_restricted"  # Harder
    SYMMETRIC = "symmetric"          # Requires relay


@dataclass
class PeerConnectivity:
    """Peer connectivity info."""
    peer_id: str
    public_addr: Optional[str]
    private_addrs: List[str]
    nat_type: NATType
    relay_addrs: List[str]  # Fallback relay addresses
    

class Libp2pTransport:
    """
    libp2p-based transport layer with NAT traversal.
    
    Connection priority:
    1. Direct (if both peers have public IPs)
    2. Hole punch (if NAT types are compatible)
    3. Relay (fallback, higher latency)
    """
    
    HOLE_PUNCH_TIMEOUT = 5.0  # seconds
    
    def __init__(
        self,
        node_id: str,
        bootstrap_peers: List[str],
        relay_servers: List[str],
    ):
        self.node_id = node_id
        self.bootstrap_peers = bootstrap_peers
        self.relay_servers = relay_servers
        
        self.host = None  # libp2p Host
        self.nat_type = NATType.UNKNOWN
        self.public_addr: Optional[str] = None
        
    async def start(self):
        """Initialize libp2p host with AutoNAT and relay."""
        # Create libp2p host with:
        # - TCP + QUIC transports
        # - Noise encryption
        # - Yamux multiplexing
        # - AutoNAT service
        # - Circuit relay client
        
        # Run AutoNAT to detect NAT type
        self.nat_type = await self._detect_nat_type()
        logger.info(f"Detected NAT type: {self.nat_type.value}")
        
        if self.nat_type == NATType.SYMMETRIC:
            logger.warning("Symmetric NAT detected - will rely on relay servers")
            await self._register_with_relays()
            
    async def connect_to_peer(self, peer_id: str, addrs: List[str]) -> 'Connection':
        """
        Connect to peer with automatic NAT traversal.
        
        Returns:
            Connection object for gRPC-like communication
        """
        # Try strategies in order
        strategies = [
            self._try_direct_connect,
            self._try_hole_punch,
            self._try_relay_connect,
        ]
        
        for strategy in strategies:
            try:
                conn = await asyncio.wait_for(
                    strategy(peer_id, addrs),
                    timeout=self.HOLE_PUNCH_TIMEOUT
                )
                if conn:
                    logger.info(f"Connected to {peer_id[:8]} via {strategy.__name__}")
                    return conn
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"{strategy.__name__} failed: {e}")
                continue
                
        raise ConnectionError(f"All NAT traversal strategies failed for {peer_id}")
    
    async def _try_direct_connect(self, peer_id: str, addrs: List[str]) -> Optional['Connection']:
        """Try direct connection (works if peer has public IP)."""
        for addr in addrs:
            try:
                return await self.host.connect(peer_id, addr)
            except:
                continue
        return None
        
    async def _try_hole_punch(self, peer_id: str, addrs: List[str]) -> Optional['Connection']:
        """
        Attempt hole punching.
        
        Requires a coordination server to sync timing:
        1. Both peers send "SYN" to coordinator
        2. Coordinator tells both to send UDP packet simultaneously
        3. NAT creates mapping, packets cross
        """
        # Coordinate via DHT or relay
        # Exchange external addresses
        # Simultaneous connect attempt
        pass
        
    async def _try_relay_connect(self, peer_id: str, addrs: List[str]) -> Optional['Connection']:
        """
        Connect via circuit relay.
        
        Higher latency but always works.
        Good for symmetric NAT or firewall situations.
        """
        for relay in self.relay_servers:
            try:
                # Connect to relay, request circuit to peer
                return await self.host.connect_via_relay(relay, peer_id)
            except:
                continue
        return None
```

**Option B: STUN/TURN Server (Simpler, Centralized)**
```python
"""
STUN/TURN based NAT traversal.

Simpler than libp2p but requires centralized infrastructure.
- STUN: Discover public IP/port, attempt hole punch
- TURN: Relay fallback when hole punch fails
"""
import socket
import struct

class STUNClient:
    """
    STUN client for public IP discovery.
    
    STUN servers:
    - stun.l.google.com:19302 (free)
    - stun.neuroshard.io:3478 (our own)
    """
    
    STUN_SERVERS = [
        ("stun.l.google.com", 19302),
        ("stun.cloudflare.com", 3478),
    ]
    
    def __init__(self):
        self.public_ip: Optional[str] = None
        self.public_port: Optional[int] = None
        
    async def discover_public_address(self) -> tuple[str, int]:
        """
        Query STUN server to discover public IP and port.
        
        Returns:
            (public_ip, public_port) tuple
        """
        for server, port in self.STUN_SERVERS:
            try:
                result = await self._query_stun(server, port)
                if result:
                    self.public_ip, self.public_port = result
                    return result
            except Exception as e:
                logger.debug(f"STUN query to {server} failed: {e}")
                continue
                
        raise RuntimeError("Could not discover public address via STUN")
        
    async def _query_stun(self, server: str, port: int) -> Optional[tuple[str, int]]:
        """Send STUN binding request and parse response."""
        # Build STUN Binding Request
        # Send UDP packet
        # Parse Binding Response for XOR-MAPPED-ADDRESS
        pass


class NATHolePuncher:
    """
    Coordinate hole punching between two NATed peers.
    
    Protocol:
    1. Both peers register with NeuroShard tracker
    2. When A wants to connect to B:
       a. A asks tracker for B's public addr
       b. Tracker tells B that A wants to connect
       c. Both start sending UDP packets to each other's public addr
       d. NAT mappings are created, connection established
    """
    
    PUNCH_ATTEMPTS = 10
    PUNCH_INTERVAL = 0.1  # 100ms between attempts
    
    def __init__(self, stun_client: STUNClient, tracker_url: str):
        self.stun = stun_client
        self.tracker_url = tracker_url
        self.local_socket: Optional[socket.socket] = None
        
    async def punch_hole(self, target_peer_id: str) -> tuple[str, int]:
        """
        Attempt to punch hole to target peer.
        
        Returns:
            (peer_ip, peer_port) for direct UDP communication
        """
        # 1. Ensure we know our public addr
        our_public = await self.stun.discover_public_address()
        
        # 2. Register with tracker, request peer info
        peer_info = await self._request_peer_from_tracker(target_peer_id, our_public)
        
        # 3. Start sending UDP packets while peer does the same
        for i in range(self.PUNCH_ATTEMPTS):
            await self._send_punch_packet(peer_info.public_addr, peer_info.public_port)
            
            # Check if we received their punch packet
            received = await self._check_incoming(timeout=self.PUNCH_INTERVAL)
            if received:
                return peer_info.public_addr, peer_info.public_port
                
        raise ConnectionError(f"Hole punch failed after {self.PUNCH_ATTEMPTS} attempts")


class NATTraversalManager:
    """
    Unified NAT traversal manager.
    
    Combines STUN discovery, hole punching, and TURN fallback.
    """
    
    def __init__(
        self,
        node_id: str,
        stun_servers: List[str],
        turn_server: Optional[str] = None,
        turn_credentials: Optional[tuple[str, str]] = None,
    ):
        self.node_id = node_id
        self.stun = STUNClient()
        self.stun.STUN_SERVERS = [(s.split(':')[0], int(s.split(':')[1])) for s in stun_servers]
        
        self.turn_server = turn_server
        self.turn_credentials = turn_credentials
        
        # Connection cache
        self.established_connections: Dict[str, tuple[str, int]] = {}
        
        # Metrics
        self.direct_success = 0
        self.holepunch_success = 0
        self.relay_success = 0
        self.total_attempts = 0
        
    @property
    def connection_success_rate(self) -> float:
        """Calculate overall connection success rate."""
        if self.total_attempts == 0:
            return 0.0
        total_success = self.direct_success + self.holepunch_success + self.relay_success
        return total_success / self.total_attempts
        
    async def connect(self, peer_id: str, peer_addrs: List[str]) -> 'Connection':
        """
        Connect to peer using best available method.
        
        Priority:
        1. Cached connection
        2. Direct (if peer has public addr)
        3. Hole punch
        4. TURN relay (fallback)
        """
        self.total_attempts += 1
        
        # Check cache
        if peer_id in self.established_connections:
            addr = self.established_connections[peer_id]
            # Verify still valid...
            
        # Try methods in order
        # ...
```

### Integration Points

| Component | Change |
|-----------|--------|
| `runner.py` | Replace `_forward_distributed()` with swarm router |
| `grpc_server.py` | Add `SwarmForward` RPC for batch activations |
| `p2p.py` | Integrate SwarmHeartbeatService |

---

## Epic 2: Asynchronous Micro-Batch Engine

### Objective
Eliminate GPU starvation by decoupling compute from communication.

**Key Directive**: "The GPU must never wait for network packets. Forward pass can run 100 steps ahead of backward if necessary."

### Architecture Target

```
┌─────────────────────────────────────────────────────────────────┐
│ TARGET: Async Pipeline (Decoupled)                              │
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │ InboundQueue│ ──→ │ GPU Compute │ ──→ │OutboundQueue│      │
│   │ (Priority)  │     │ (Saturated) │     │ (Async Send)│      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│         ↑                                        │              │
│         │                                        ↓              │
│   ┌─────────────┐                        ┌─────────────┐       │
│   │ Network Rx  │←─────────────────────→│ Network Tx  │        │
│   │ (Async)     │                        │ (Async)     │        │
│   └─────────────┘                        └─────────────┘       │
│                                                                 │
│   METRIC: Buffer Fill Rate (empty=starved, full=good)          │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Tasks

#### Task 2.1: Activation Buffer System
**File**: `neuroshard/core/activation_buffer.py` (NEW)

```python
import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import IntEnum


class ActivationPriority(IntEnum):
    """Priority levels for activation processing."""
    INFERENCE_URGENT = 0   # Paid inference, user waiting
    INFERENCE_NORMAL = 10  # Standard inference
    TRAINING_FORWARD = 20  # Training forward pass
    TRAINING_BACKWARD = 30 # Training backward pass (can be delayed)


@dataclass(order=True)
class ActivationPacket:
    """A unit of work for the compute engine."""
    priority: int
    timestamp: float = field(compare=False)
    
    # Payload
    session_id: str = field(compare=False)
    micro_batch_id: int = field(compare=False)
    tensor_data: torch.Tensor = field(compare=False)
    
    # Routing
    source_node: str = field(compare=False)
    target_layer: int = field(compare=False)
    is_backward: bool = field(compare=False, default=False)
    
    # For backward pass
    requires_grad: bool = field(compare=False, default=False)
    grad_output: Optional[torch.Tensor] = field(compare=False, default=None)


class ActivationBuffer:
    """
    Priority queue for incoming activations.
    
    Decouples network I/O from GPU computation.
    GPU thread simply pops next item, computes, pushes to outbound.
    
    Metrics:
    - fill_rate: 0.0 = starved (bad), 1.0 = full (backpressured)
    - avg_wait_time: How long packets wait before processing
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._queue: List[ActivationPacket] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        # Metrics
        self.packets_in = 0
        self.packets_out = 0
        self.total_wait_time = 0.0
        
    @property
    def fill_rate(self) -> float:
        """Current fill rate (0.0 to 1.0)."""
        with self._lock:
            return len(self._queue) / self.max_size
    
    def put(self, packet: ActivationPacket, timeout: float = None) -> bool:
        """
        Add activation to buffer.
        
        Returns False if buffer full and timeout expires.
        """
        with self._not_full:
            if len(self._queue) >= self.max_size:
                if not self._not_full.wait(timeout):
                    return False  # Timeout, buffer still full
            
            heapq.heappush(self._queue, packet)
            self.packets_in += 1
            self._not_empty.notify()
            return True
    
    def get(self, timeout: float = None) -> Optional[ActivationPacket]:
        """
        Get highest-priority activation.
        
        Returns None if buffer empty and timeout expires.
        """
        with self._not_empty:
            if not self._queue:
                if not self._not_empty.wait(timeout):
                    return None  # Timeout, buffer still empty
            
            packet = heapq.heappop(self._queue)
            wait_time = time.time() - packet.timestamp
            self.total_wait_time += wait_time
            self.packets_out += 1
            self._not_full.notify()
            return packet
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        with self._lock:
            avg_wait = self.total_wait_time / max(1, self.packets_out)
            return {
                "fill_rate": self.fill_rate,
                "queue_size": len(self._queue),
                "packets_in": self.packets_in,
                "packets_out": self.packets_out,
                "avg_wait_time_ms": avg_wait * 1000,
                "is_starved": self.fill_rate < 0.1,
                "is_backpressured": self.fill_rate > 0.9,
            }


class OutboundBuffer:
    """Buffer for outgoing activations (async network send)."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.send_in_progress = 0
        
    async def put(self, packet: ActivationPacket):
        """Queue packet for sending."""
        await self._queue.put(packet)
        
    async def send_loop(self, swarm_router: 'SwarmRouter'):
        """Continuously send packets to peers."""
        while True:
            packet = await self._queue.get()
            self.send_in_progress += 1
            
            try:
                await swarm_router.send_with_failover(
                    packet.tensor_data,
                    packet.target_layer,
                    packet.session_id
                )
            except Exception as e:
                logger.error(f"Failed to send activation: {e}")
            finally:
                self.send_in_progress -= 1
                self._queue.task_done()
```

#### Task 2.2: Compute Engine with Soft Overflow (Decoupled Worker)
**File**: `neuroshard/core/compute_engine.py` (NEW)

**Context**: If OutboundBuffer is full (network congestion), the ComputeEngine must NOT wait. Waiting causes GPU starvation, defeating the purpose of async design.

**Key Directive**: "If outbound.full(): Do not await. Instead: accumulate_gradient_locally() and discard the activation (skip sending). Treat it as a DiLoCo-style local-only training step."

**Metric**: `local_only_steps / total_steps` (should be < 5% under normal conditions)

```python
from enum import Enum
from typing import Dict, Optional
import torch
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class StepOutcome(Enum):
    """Outcome of a compute step."""
    SENT = "sent"               # Normal: activation sent to next peer
    LOCAL_ONLY = "local_only"   # Soft overflow: accumulated locally, activation discarded
    DROPPED = "dropped"         # Critical overflow: couldn't even accumulate


class ComputeEngine:
    """
    Decoupled GPU compute worker with Soft Overflow handling.
    
    Pulls from inbound buffer, computes, pushes to outbound.
    
    CRITICAL: Never waits for network - GPU must never stall!
    
    Soft Overflow Logic (The "Don't Stop" Mechanism):
    ================================================
    When outbound buffer is full (network congestion):
    
    1. DO NOT await outbound.put() - this would stall the GPU
    2. Instead, accumulate gradients locally (DiLoCo style)
    3. Discard the activation (don't try to send it)
    4. Continue processing next packet
    
    Rationale: Better to treat a step as "local-only training" than to
    halt the GPU waiting for the network. The DiLoCo outer optimizer
    will eventually sync these local updates.
    
    Supports Interleaved 1F1B schedule:
    - Process multiple forward micro-batches
    - Interleave with backward passes
    - Hides backward pass latency with forward work
    """
    
    # Soft overflow thresholds
    OUTBOUND_SOFT_LIMIT = 0.9    # Start soft overflow at 90% full
    OUTBOUND_HARD_LIMIT = 0.99   # Hard limit - must discard
    
    def __init__(
        self,
        model: 'DynamicNeuroLLM',
        inbound: 'ActivationBuffer',
        outbound: 'OutboundBuffer',
        diloco_trainer: Optional['DiLoCoTrainer'] = None,
        num_micro_batches: int = 4
    ):
        self.model = model
        self.inbound = inbound
        self.outbound = outbound
        self.diloco = diloco_trainer  # For local gradient accumulation
        self.num_micro_batches = num_micro_batches
        
        # Interleaved 1F1B state
        self.pending_backwards: Dict[int, 'ActivationPacket'] = {}
        self.forward_count = 0
        self.backward_count = 0
        
        # Soft overflow state
        self.local_gradient_buffer: Dict[str, torch.Tensor] = {}
        self.local_only_steps = 0
        self.dropped_steps = 0
        self.total_steps = 0
        
        self.running = False
        
    @property
    def local_only_rate(self) -> float:
        """Fraction of steps that were local-only due to overflow."""
        if self.total_steps == 0:
            return 0.0
        return self.local_only_steps / self.total_steps
        
    @property
    def drop_rate(self) -> float:
        """Fraction of steps that were completely dropped."""
        if self.total_steps == 0:
            return 0.0
        return self.dropped_steps / self.total_steps
        
    def _check_outbound_pressure(self) -> str:
        """
        Check outbound buffer pressure level.
        
        Returns:
            "ok" - can send normally
            "soft_overflow" - buffer almost full, use local accumulation
            "hard_overflow" - buffer completely full, must discard
        """
        fill_rate = self.outbound.fill_rate
        
        if fill_rate >= self.OUTBOUND_HARD_LIMIT:
            return "hard_overflow"
        elif fill_rate >= self.OUTBOUND_SOFT_LIMIT:
            return "soft_overflow"
        else:
            return "ok"
        
    async def run(self):
        """
        Main compute loop with Interleaved 1F1B schedule and Soft Overflow.
        
        Schedule (for 4 micro-batches):
        F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...
        
        Key insight: Start backward passes BEFORE all forwards complete.
        This overlaps backward compute with forward network latency.
        
        SOFT OVERFLOW: If network is congested, steps become local-only.
        """
        self.running = True
        
        while self.running:
            # Check inbound buffer
            packet = self.inbound.get(timeout=0.01)
            
            if packet is None:
                # Buffer empty - GPU starved!
                await asyncio.sleep(0.001)
                continue
            
            self.total_steps += 1
            
            if packet.is_backward:
                # Process backward pass
                outcome = await self._process_backward(packet)
            else:
                # Process forward pass with soft overflow handling
                outcome = await self._process_forward_with_overflow(packet)
            
            # Track outcomes
            if outcome == StepOutcome.LOCAL_ONLY:
                self.local_only_steps += 1
            elif outcome == StepOutcome.DROPPED:
                self.dropped_steps += 1
            
            # Interleaved 1F1B: After warmup, alternate F-B
            if self.forward_count > self.num_micro_batches:
                # Check if we should do a backward
                if self.pending_backwards:
                    oldest_mb = min(self.pending_backwards.keys())
                    bp = self.pending_backwards.pop(oldest_mb)
                    await self._process_backward(bp)
                    
            # Log overflow statistics periodically
            if self.total_steps % 100 == 0:
                logger.info(
                    f"ComputeEngine stats: total={self.total_steps}, "
                    f"local_only_rate={self.local_only_rate:.2%}, "
                    f"drop_rate={self.drop_rate:.2%}"
                )
    
    async def _process_forward_with_overflow(self, packet: 'ActivationPacket') -> StepOutcome:
        """
        Process forward pass with soft overflow handling.
        
        The "Don't Stop" Logic:
        1. Always compute forward pass (GPU never waits)
        2. Check outbound pressure
        3. If congested: accumulate locally, skip sending
        4. If ok: queue for outbound
        """
        # ALWAYS compute forward - GPU must never stall
        output = self.model.forward_my_layers(
            packet.tensor_data.to(self.model.device)
        )
        self.forward_count += 1
        
        # Check backpressure AFTER compute
        pressure = self._check_outbound_pressure()
        
        if pressure == "ok":
            # Normal path: queue activation for sending
            next_layer = max(self.model.my_layer_ids) + 1
            outbound_packet = ActivationPacket(
                priority=packet.priority,
                timestamp=time.time(),
                session_id=packet.session_id,
                micro_batch_id=packet.micro_batch_id,
                tensor_data=output.cpu(),
                source_node=self.model.node_id,
                target_layer=next_layer,
            )
            
            # Non-blocking put with timeout
            try:
                await asyncio.wait_for(
                    self.outbound.put(outbound_packet),
                    timeout=0.01  # 10ms max wait
                )
                return StepOutcome.SENT
            except asyncio.TimeoutError:
                # Couldn't send in time - fall through to local accumulation
                pressure = "soft_overflow"
        
        if pressure == "soft_overflow":
            # SOFT OVERFLOW: Network congested
            # Treat as local-only DiLoCo step
            logger.debug(
                f"Soft overflow at step {self.total_steps}: "
                f"accumulating locally, discarding activation"
            )
            
            # Accumulate gradient locally (if training)
            if packet.requires_grad and self.diloco is not None:
                self._accumulate_local_gradient(output, packet)
            
            # Discard activation - don't try to send
            del output
            
            return StepOutcome.LOCAL_ONLY
            
        else:  # hard_overflow
            # HARD OVERFLOW: Critical congestion
            # Cannot even accumulate - must drop
            logger.warning(
                f"Hard overflow at step {self.total_steps}: "
                f"dropping step entirely (outbound at {self.outbound.fill_rate:.1%})"
            )
            
            del output
            return StepOutcome.DROPPED
    
    def _accumulate_local_gradient(self, output: torch.Tensor, packet: 'ActivationPacket'):
        """
        Accumulate gradient locally during soft overflow.
        
        This implements the DiLoCo "local training" behavior:
        - Compute gradient for this step
        - Add to local accumulator
        - Will be synced during next DiLoCo outer step
        """
        if not self.model.training:
            return
            
        # If we have the upstream gradient, compute local gradient
        if packet.grad_output is not None:
            # Backward through our layers
            output.backward(packet.grad_output)
            
            # Accumulate into DiLoCo trainer
            if self.diloco:
                self.diloco.inner_step_count += 1
        else:
            # For forward-only steps during overflow, we'll catch up
            # when network recovers. The key is: GPU keeps running.
            pass
    
    async def _process_backward(self, packet: 'ActivationPacket') -> StepOutcome:
        """
        Process backward activation packet.
        
        Backward passes are always processed - they contain gradients
        that must be applied.
        """
        # Retrieve saved activations for this micro-batch
        # Compute gradients
        # Queue gradient to outbound (for gradient gossip)
        self.backward_count += 1
        
        # Backward passes also respect soft overflow for gradient sending
        pressure = self._check_outbound_pressure()
        
        if pressure != "ok":
            # Accumulate gradient locally instead of gossiping
            logger.debug(f"Backward pass local accumulation at step {self.total_steps}")
            return StepOutcome.LOCAL_ONLY
            
        return StepOutcome.SENT
    
    def get_stats(self) -> Dict[str, float]:
        """Get compute engine statistics."""
        return {
            "total_steps": self.total_steps,
            "forward_count": self.forward_count,
            "backward_count": self.backward_count,
            "local_only_steps": self.local_only_steps,
            "dropped_steps": self.dropped_steps,
            "local_only_rate": self.local_only_rate,
            "drop_rate": self.drop_rate,
            "outbound_fill_rate": self.outbound.fill_rate,
            "inbound_fill_rate": self.inbound.fill_rate,
        }
```

#### Task 2.3: Dynamic Micro-Batch Sizing
**File**: `neuroshard/core/micro_batch_config.py` (NEW)

```python
@dataclass
class MicroBatchConfig:
    """
    Configuration for micro-batch sizing.
    
    "Goldilocks Zone" - large enough to saturate GPU, small enough for network MTU.
    """
    # Base sizes
    seq_len: int = 512
    micro_batch_size: int = 4
    
    # Activation size estimation
    hidden_dim: int = 768
    bytes_per_element: int = 4  # FP32
    
    # Network constraints
    target_packet_size_kb: int = 64  # Target ~64KB per packet
    max_packet_size_kb: int = 256    # Hard limit (network MTU)
    
    def estimate_activation_size_kb(self) -> float:
        """Estimate activation tensor size in KB."""
        elements = self.micro_batch_size * self.seq_len * self.hidden_dim
        return (elements * self.bytes_per_element) / 1024
    
    def auto_tune(self, available_memory_mb: float, network_bandwidth_mbps: float):
        """Auto-tune micro-batch size based on hardware."""
        # Start with max that fits in memory
        max_by_memory = int(available_memory_mb * 0.1 / self.estimate_activation_size_kb() * self.micro_batch_size)
        
        # Constrain by network
        max_by_network = int(self.target_packet_size_kb * 1024 / 
                            (self.seq_len * self.hidden_dim * self.bytes_per_element))
        
        self.micro_batch_size = max(1, min(max_by_memory, max_by_network, 16))
        
        logger.info(f"Auto-tuned micro_batch_size={self.micro_batch_size} "
                   f"(memory limit: {max_by_memory}, network limit: {max_by_network})")
```

### Integration Points

| Component | Change |
|-----------|--------|
| `runner.py` | Replace blocking `train_step()` with async engine |
| `grpc_server.py` | Push to ActivationBuffer instead of direct process |
| `dynamic_model.py` | DynamicNeuroNode uses ComputeEngine |

---

## Epic 3: Extreme Gradient Accumulation & Verification

### Objective
Slash bandwidth by 90%+ through DiLoCo-style lazy syncing.

**Key Directive**: "Accumulate locally for as long as possible. Sync only when statistically necessary."

### Architecture Target

```
┌─────────────────────────────────────────────────────────────────┐
│ TARGET: DiLoCo-Style Lazy Syncing                               │
│                                                                 │
│   Node A: [step 1] [step 2] ... [step N] ──→ SYNC              │
│   Node B: [step 1] [step 2] ... [step N] ──→ SYNC              │
│   Node C: [step 1] [step 2] ... [step N] ──→ SYNC              │
│                                                │                │
│                                                ↓                │
│                                    [Robust Aggregation]         │
│                                                │                │
│                                                ↓                │
│                                         [All Apply]             │
│                                                                 │
│   SYNC INTERVAL: Every N steps (e.g., 500)                     │
│   BANDWIDTH: Reduced by N×                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Tasks

#### Task 3.1: DiLoCo-Style Local Accumulation
**File**: `neuroshard/core/diloco_trainer.py` (NEW)

```python
class DiLoCoTrainer:
    """
    Distributed Local SGD with Outer Optimizer (DiLoCo).
    
    Based on: "DiLoCo: Distributed Low-Communication Training of Language Models"
    
    Key idea:
    - Each node trains independently for N steps (inner loop)
    - Periodically sync pseudo-gradients (outer loop)
    - Outer optimizer (e.g., Nesterov momentum) on delta
    
    Benefits:
    - N× reduction in communication
    - More robust to stragglers
    - Better for high-latency networks
    """
    
    DEFAULT_INNER_STEPS = 500
    DEFAULT_OUTER_LR = 0.7
    DEFAULT_OUTER_MOMENTUM = 0.9
    
    def __init__(
        self,
        model: DynamicNeuroLLM,
        inner_optimizer: torch.optim.Optimizer,
        inner_steps: int = DEFAULT_INNER_STEPS,
        outer_lr: float = DEFAULT_OUTER_LR,
        outer_momentum: float = DEFAULT_OUTER_MOMENTUM,
    ):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        
        # Store initial weights at start of inner loop
        self.initial_weights: Dict[str, torch.Tensor] = {}
        
        # Outer optimizer momentum buffer
        self.outer_momentum_buffer: Dict[str, torch.Tensor] = {}
        
        # Tracking
        self.inner_step_count = 0
        self.outer_step_count = 0
        
    def save_initial_weights(self):
        """Save weights at start of inner loop."""
        self.initial_weights = {
            name: param.data.clone()
            for name, param in self.model.parameters()
        }
        self.inner_step_count = 0
        
    def inner_step(self, loss: torch.Tensor):
        """
        Execute one inner optimization step.
        
        Just normal SGD/AdamW - no communication.
        """
        self.inner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.inner_optimizer.step()
        
        self.inner_step_count += 1
        
    def should_sync(self) -> bool:
        """Check if we should trigger outer sync."""
        return self.inner_step_count >= self.inner_steps
    
    def compute_pseudo_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Compute pseudo-gradient (delta from initial weights).
        
        This is what gets communicated to peers.
        """
        pseudo_grads = {}
        
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                # Pseudo-gradient = initial - current (direction of improvement)
                delta = self.initial_weights[name] - param.data
                pseudo_grads[name] = delta
        
        return pseudo_grads
    
    def apply_outer_update(self, aggregated_pseudo_grads: Dict[str, torch.Tensor]):
        """
        Apply outer optimizer step with aggregated pseudo-gradients.
        
        Uses Nesterov momentum for stability.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in aggregated_pseudo_grads:
                    continue
                    
                delta = aggregated_pseudo_grads[name]
                
                # Nesterov momentum update
                if name not in self.outer_momentum_buffer:
                    self.outer_momentum_buffer[name] = torch.zeros_like(delta)
                
                momentum = self.outer_momentum_buffer[name]
                momentum.mul_(self.outer_momentum).add_(delta)
                
                # Apply update: w = w + lr * (momentum + delta)
                update = self.outer_lr * (self.outer_momentum * momentum + delta)
                param.data.add_(update)
                
                self.outer_momentum_buffer[name] = momentum
        
        self.outer_step_count += 1
        self.save_initial_weights()  # Reset for next inner loop
        
        logger.info(f"DiLoCo outer step {self.outer_step_count} complete "
                   f"({self.inner_steps} inner steps accumulated)")
```

#### Task 3.2: Speculative Checkpointing
**File**: `neuroshard/core/speculative_checkpoint.py` (NEW)

```python
class SpeculativeCheckpointer:
    """
    High-frequency snapshots for fast crash recovery.
    
    Runs in background thread, saves every 2 minutes.
    On crash, neighbors can fetch "hot" snapshot.
    
    Key insight: Cheaper to over-checkpoint than to re-train.
    """
    
    SNAPSHOT_INTERVAL = 120  # 2 minutes
    MAX_SNAPSHOTS = 5        # Keep last 5
    
    def __init__(
        self,
        model: DynamicNeuroLLM,
        optimizer: torch.optim.Optimizer,
        diloco_trainer: DiLoCoTrainer,
        checkpoint_dir: Path,
        p2p_manager: 'P2PManager'
    ):
        self.model = model
        self.optimizer = optimizer
        self.diloco = diloco_trainer
        self.checkpoint_dir = checkpoint_dir
        self.p2p = p2p_manager
        
        self.running = False
        self.snapshots: List[Path] = []
        
    def start(self):
        """Start background checkpointing."""
        self.running = True
        threading.Thread(target=self._checkpoint_loop, daemon=True).start()
        
    def _checkpoint_loop(self):
        """Continuous checkpointing in background."""
        while self.running:
            time.sleep(self.SNAPSHOT_INTERVAL)
            
            try:
                snapshot_path = self._save_snapshot()
                self._announce_snapshot(snapshot_path)
                self._cleanup_old_snapshots()
            except Exception as e:
                logger.error(f"Speculative checkpoint failed: {e}")
    
    def _save_snapshot(self) -> Path:
        """Save current state as hot snapshot."""
        timestamp = int(time.time())
        path = self.checkpoint_dir / f"hot_snapshot_{timestamp}.pt"
        
        torch.save({
            "model_state": {
                name: param.data.clone() 
                for name, param in self.model.named_parameters()
            },
            "optimizer_state": self.optimizer.state_dict(),
            "diloco_state": {
                "initial_weights": self.diloco.initial_weights,
                "outer_momentum": self.diloco.outer_momentum_buffer,
                "inner_step_count": self.diloco.inner_step_count,
                "outer_step_count": self.diloco.outer_step_count,
            },
            "timestamp": timestamp,
            "node_id": self.model.node_id,
            "layer_ids": self.model.my_layer_ids,
        }, path)
        
        self.snapshots.append(path)
        logger.info(f"Hot snapshot saved: {path.name}")
        
        return path
    
    def _announce_snapshot(self, path: Path):
        """Announce snapshot availability to DHT."""
        if self.p2p and self.p2p.dht:
            key = f"snapshot_{self.model.node_id}"
            self.p2p.dht.store(key, str(path))
    
    def _cleanup_old_snapshots(self):
        """Remove old snapshots, keep only MAX_SNAPSHOTS."""
        while len(self.snapshots) > self.MAX_SNAPSHOTS:
            old = self.snapshots.pop(0)
            try:
                old.unlink()
            except FileNotFoundError:
                pass
    
    def fetch_neighbor_snapshot(self, neighbor_id: str) -> Optional[Dict]:
        """
        Fetch hot snapshot from a neighbor.
        
        Used for fast recovery after crash.
        """
        if not self.p2p or not self.p2p.dht:
            return None
            
        key = f"snapshot_{neighbor_id}"
        snapshot_path = self.p2p.dht.lookup_value(key)
        
        if not snapshot_path:
            return None
        
        # Request via gRPC (implementation in grpc_server.py)
        # ...
```

#### Task 3.3: Enhanced Verification for Accumulated Gradients
**File**: `neuroshard/core/robust_aggregation.py` (MODIFY)

```python
# Add to existing RobustAggregator class:

class AccumulatedGradientVerifier:
    """
    Verify accumulated gradients are statistically sound.
    
    Since DiLoCo syncs less often, bad gradients are more damaging.
    Enhanced verification required.
    """
    
    def __init__(self, trusted_batch_size: int = 16):
        self.trusted_batch_size = trusted_batch_size
        
    def verify_accumulated_gradient(
        self,
        submitted_grad: Dict[str, torch.Tensor],
        local_model: DynamicNeuroLLM,
        local_data_sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[bool, str]:
        """
        Verify submitted gradient against local trusted micro-batch.
        
        1. Compute gradient on local trusted data
        2. Compare distribution to submitted gradient
        3. Reject if significantly different
        
        Returns (is_valid, reason)
        """
        # Compute local "trusted" gradient
        input_ids, labels = local_data_sample
        
        local_model.train()
        outputs = local_model.forward_my_layers(
            local_model.model.embed(input_ids)
        )
        
        if local_model.model.has_lm_head:
            logits = local_model.model.compute_logits(outputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
        
        # Extract trusted gradient
        trusted_grad = {
            name: param.grad.clone()
            for name, param in local_model.named_parameters()
            if param.grad is not None
        }
        
        # Compare distributions
        for name, submitted in submitted_grad.items():
            if name not in trusted_grad:
                continue
                
            trusted = trusted_grad[name]
            
            # Check 1: Cosine similarity (direction)
            cosine_sim = F.cosine_similarity(
                submitted.flatten().unsqueeze(0),
                trusted.flatten().unsqueeze(0)
            ).item()
            
            if cosine_sim < 0.5:  # Threshold: at least 50% aligned
                return False, f"Gradient direction mismatch for {name}: cosine={cosine_sim:.3f}"
            
            # Check 2: Magnitude ratio
            submitted_norm = submitted.norm().item()
            trusted_norm = trusted.norm().item()
            
            if trusted_norm > 0:
                ratio = submitted_norm / trusted_norm
                if ratio > 10 or ratio < 0.1:  # 10x tolerance
                    return False, f"Gradient magnitude mismatch for {name}: ratio={ratio:.2f}"
            
            # Check 3: Variance ratio (Levene's test approximation)
            submitted_var = submitted.var().item()
            trusted_var = trusted.var().item()
            
            if trusted_var > 0:
                var_ratio = submitted_var / trusted_var
                if var_ratio > 100 or var_ratio < 0.01:
                    return False, f"Gradient variance mismatch for {name}: ratio={var_ratio:.2f}"
        
        return True, "Gradient verification passed"
```

### Integration Points

| Component | Change |
|-----------|--------|
| `runner.py` | Replace `train_step()` with DiLoCoTrainer |
| `p2p.py` | Add `_diloco_sync_loop()` for outer step gossip |
| `distributed_training.py` | TrainingCoordinator uses DiLoCoTrainer |
| `grpc_server.py` | Add `GetHotSnapshot` RPC |

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE
- [x] Implement `SwarmRouter` with basic failover
- [x] Implement `ActivationBuffer` and `OutboundBuffer`
- [x] Add capacity heartbeat protocol
- [x] Unit tests for all new components (27 tests)

### Phase 2: Integration (Weeks 3-4) ✅ COMPLETE
- [x] Integrate SwarmRouter into `dynamic_model.py`
- [x] Replace blocking forward with async engine
- [x] Update gRPC server for new message types
- [x] Integration tests with 3+ node cluster (18 tests)

### Phase 3: DiLoCo (Weeks 5-6) ✅ COMPLETE
- [x] Implement `DiLoCoTrainer` with Nesterov outer optimizer
- [x] Implement `SpeculativeCheckpointer` with hot/cold snapshots
- [x] Enhanced gradient verification (RobustAggregator)
- [x] Full system tests with simulated network delays (44 tests)

### Phase 4: Runner Integration (Weeks 7-8) ✅ COMPLETE

**Goal**: Integrate Swarm components into `runner.py` - Swarm IS the architecture (no toggles!)

#### Design Philosophy
The Swarm architecture is the ONLY architecture. There are no:
- `--swarm` flags to enable/disable
- `enable_swarm` config options
- `is_swarm_node` conditional checks
- Fallbacks to "non-swarm" mode

Every node runs the full swarm stack.

#### Task 4.1: SwarmEnabledNode Integration
- [x] `create_swarm_node()` replaces `create_dynamic_node()` as the default
- [x] Initialize `SwarmRouter`, `ActivationBuffer`, `OutboundBuffer` automatically
- [x] Start `SwarmHeartbeatService` alongside P2P

#### Task 4.2: Async Training Loop
- [x] `DiLoCoTrainer` integrated for lazy gradient sync (default: 500 steps)
- [x] `SpeculativeCheckpointer` for crash recovery (snapshots every 2 min)
- [x] `RobustAggregator` with configurable strategies (trimmed_mean default)

#### Task 4.3: Swarm gRPC Integration  
- [x] `SwarmForward`, `GetSwarmStatus`, `UpdatePeerCapacity` RPCs
- [x] TCP fallback for UDP heartbeat failures
- [x] No conditional checks - all nodes support swarm RPCs

#### Task 4.4: P2P Updates
- [x] Add `get_swarm_status()`, `get_diloco_progress()`, `get_network_health()`
- [x] P2PManager integrates with swarm components

#### Task 4.5: Logging Improvements
- [x] `SwarmLogger` with role-specific prefixes (DRIVER, WORKER, VALIDATOR)
- [x] JSON-format logging option
- [x] Periodic summary stats every 60s (reduce log spam)

#### Task 4.6: API Endpoints
- [x] `GET /api/swarm` - buffer fill rates, router stats
- [x] `GET /api/diloco` - inner step count, sync progress
- [x] DiLoCo progress included in `GET /api/stats`

### Phase 5: Optimization & Production (Future)
- [ ] Performance profiling and bottleneck analysis
- [ ] Auto-tuning for micro-batch size
- [ ] Memory optimization for large models
- [ ] Production hardening and monitoring
- [ ] GUI updates for swarm visualization

---

## Metrics & Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| GPU Utilization | ~40% (estimated) | **95%** |
| Network Bandwidth per Step | 100% (every step) | **~0.2%** (every 500 steps) |
| Mean Time to Failover | N/A (crash) | **<200ms** |
| Buffer Fill Rate | N/A | **0.5-0.9** (healthy) |
| Checkpoint Recovery Time | Full restart | **<30s** (hot snapshot) |

---

## References

1. **SWARM Parallelism**: Ryabinin et al., "SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient"
2. **DiLoCo**: Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models"
3. **Interleaved 1F1B**: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters"
4. **Byzantine-Robust Aggregation**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"

---

## Appendix: File Change Summary

### Phase 1 Files (Foundation)
| File | Action | Description |
|------|--------|-------------|
| `neuroshard/core/swarm_router.py` | CREATE | Multipath routing with failover |
| `neuroshard/core/swarm_heartbeat.py` | CREATE | Capacity bitmask broadcast |
| `neuroshard/core/activation_buffer.py` | CREATE | Priority queue for activations |
| `neuroshard/core/compute_engine.py` | CREATE | Async GPU worker |
| `neuroshard/core/nat_traversal.py` | CREATE | NAT traversal (STUN/TURN) |

### Phase 2 Files (Integration)
| File | Action | Description |
|------|--------|-------------|
| `neuroshard/core/swarm_service.py` | CREATE | gRPC service mixin for swarm |
| `neuroshard/core/swarm_integration.py` | CREATE | SwarmEnabledNode wrapper |
| `neuroshard/core/async_trainer.py` | CREATE | Async training loop |
| `protos/neuroshard.proto` | MODIFY | SwarmForward, GetSwarmStatus RPCs |

### Phase 3 Files (DiLoCo)
| File | Action | Description |
|------|--------|-------------|
| `neuroshard/core/diloco_trainer.py` | CREATE | DiLoCo local accumulation + outer optimizer |
| `neuroshard/core/speculative_checkpoint.py` | CREATE | Hot/cold snapshot system |
| `neuroshard/core/robust_aggregation.py` | CREATE | Byzantine-tolerant gradient aggregation |

### Test Files
| File | Action | Description |
|------|--------|-------------|
| `tests/test_swarm_components.py` | CREATE | Phase 1 unit tests (27 tests) |
| `tests/test_swarm_phase2.py` | CREATE | Phase 2 integration tests (18 tests) |
| `tests/test_swarm_phase3.py` | CREATE | Phase 3 DiLoCo tests (44 tests) |
| `tests/test_swarm_phase4.py` | CREATE | Phase 4 runner integration tests (24 tests) |

### Phase 4 Files (Runner Integration)
| File | Action | Description |
|------|--------|-------------|
| `neuroshard/core/swarm_factory.py` | CREATE | Factory for SwarmEnabledDynamicNode |
| `neuroshard/core/swarm_logger.py` | CREATE | Structured logging with summaries |
| `runner.py` | MODIFY | SwarmEnabledNode, --swarm flag, API endpoints |
| `neuroshard/grpc_server.py` | MODIFY | Add SwarmForward, GetSwarmStatus RPCs |
| `neuroshard/core/p2p.py` | MODIFY | Add swarm status methods |

### Documentation Updates
| File | Action | Description |
|------|--------|-------------|
| `docs/whitepaper/neuroshard_whitepaper.tex` | MODIFY | Swarm architecture section |
| `README.md` | MODIFY | Update architecture docs |

