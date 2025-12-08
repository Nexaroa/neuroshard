"""
NeuroShard Kademlia DHT Implementation Plan

1. Node ID & Distance Metric
   - SHA-1 or SHA-256 hash of (IP + Port + Random) -> 160-bit ID.
   - Distance: XOR metric.

2. Routing Table (K-Buckets)
   - List of buckets where index i contains nodes with distance 2^i to 2^(i+1).
   - K=20 (Standard Kademlia parameter).
   - Replacement cache for unresponsive nodes.

3. Protocol Messages (RPCs)
   - PING: Check liveness.
   - STORE(key, value): Store peer info or shard location.
   - FIND_NODE(target_id): Returns k closest nodes to target.
   - FIND_VALUE(key): Returns value if present, else k closest nodes.

4. Storage
   - Values are Peer Metadata: {ip, port, shard_range, last_seen}.
   - Keys: Hash of the shard_range (e.g., hash("0-4")) or Node ID.

5. Bootstrapping
   - Join network by contacting any known node (can fallback to Tracker temporarily).
   - Perform node lookup for self ID to populate buckets.

6. Integration
   - P2PManager will query DHT for "Layer X" peers instead of asking Tracker.
"""

import hashlib
import time
from typing import List, Optional

K_BUCKET_SIZE = 20
ID_LENGTH_BITS = 160

def generate_id(data: str) -> int:
    """Generate 160-bit ID."""
    return int(hashlib.sha1(data.encode()).hexdigest(), 16)

class DHTNode:
    def __init__(self, ip: str, port: int, id: Optional[int] = None):
        self.ip = ip
        self.port = port
        self.id = id if id else generate_id(f"{ip}:{port}:{time.time()}")
        self.last_seen = time.time()
    
    def distance_to(self, other_id: int) -> int:
        return self.id ^ other_id

    def __repr__(self):
        return f"<DHTNode {self.id} {self.ip}:{self.port}>"


