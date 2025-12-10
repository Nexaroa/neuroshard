import grpc
import threading
import time
import logging
from concurrent import futures
from typing import List, Optional

from protos import neuroshard_pb2
from protos import neuroshard_pb2_grpc
from neuroshard.core.network.dht import Node, RoutingTable, ID_BITS
from neuroshard.core.network.dht_service import node_to_proto, proto_to_node
from neuroshard.core.network.connection_pool import get_channel

logger = logging.getLogger("DHT")

class DHTProtocol:
    def __init__(self, local_node: Node, routing_table: RoutingTable, port: int):
        self.local_node = local_node
        self.routing_table = routing_table
        self.port = port
        # Initialize internal storage for DHT values (acting as a node)
        self.storage = {}
        # We don't manage the server here anymore, it's mixed into the main gRPC server
        # Rate limiting for "no peers" messages (log max once per 60 seconds)
        self._last_no_peers_log = {}

    def _get_stub(self, target: Node):
        # gRPC port assumption: http port + 1000
        # If target.port is already the gRPC port, we use it directly.
        # Our convention in p2p.py was: p_port = parsed.port or 80.
        # And in grpc_server.py: grpc_port = port + 1000.
        # So the node.port stored in DHT is likely the HTTP port.
        
        # Let's assume Node stores HTTP port, so we add 1000.
        # We should standardize this. For now, consistent with existing logic.
        target_addr = f"{target.ip}:{target.port + 1000}"
        channel = get_channel(target_addr)
        return neuroshard_pb2_grpc.NeuroShardServiceStub(channel)

    def ping(self, target: Node) -> bool:
        """Send PING to target node."""
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTPingRequest(
                sender=node_to_proto(self.local_node)
            )
            resp = stub.DHTPing(req, timeout=2.0)
            # Update routing table with responder
            self.routing_table.add_contact(proto_to_node(resp.responder))
            return True
        except grpc.RpcError as e:
            logger.debug(f"Ping failed to {target}: {e}")
            return False

    def find_node(self, target: Node, search_id: int) -> List[Node]:
        """Ask target for nodes closest to search_id."""
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTFindNodeRequest(
                sender=node_to_proto(self.local_node),
                target_id=search_id.to_bytes(20, byteorder='big')
            )
            resp = stub.DHTFindNode(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            
            nodes = [proto_to_node(n) for n in resp.nodes]
            return nodes
        except grpc.RpcError:
            return []

    def store(self, target: Node, key: int, value: str) -> bool:
        """Ask target to store key=value."""
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTStoreRequest(
                sender=node_to_proto(self.local_node),
                key=key.to_bytes(20, byteorder='big'),
                value=value
            )
            resp = stub.DHTStore(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            return resp.success
        except grpc.RpcError:
            return False
        
    def find_value(self, target: Node, key: int):
        """Ask target for value at key."""
        # Returns (value: str | None, nodes: List[Node])
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTFindValueRequest(
                sender=node_to_proto(self.local_node),
                key=key.to_bytes(20, byteorder='big')
            )
            resp = stub.DHTFindValue(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            
            if resp.found:
                return (resp.value, [])
            else:
                nodes = [proto_to_node(n) for n in resp.nodes]
                return (None, nodes)
        except grpc.RpcError:
            return (None, [])

    # --- High Level Lookup Algorithm (Iterative) ---
    
    def lookup_node(self, target_id: int) -> List[Node]:
        """
        Standard Kademlia Lookup
        """
        shortlist = self.routing_table.find_closest(target_id)
        if not shortlist:
            return []
            
        visited = set()
        visited.add(self.local_node.id) # Don't query self
        
        active_queries = 0
        alpha = 3
        
        # Iterative lookup
        converged = False
        while not converged:
            candidates = [n for n in shortlist if n.id not in visited][:alpha]
            if not candidates:
                break
                
            results_found = False
            for node in candidates:
                visited.add(node.id)
                
                new_nodes = self.find_node(node, target_id)
                if new_nodes:
                    results_found = True
                    for n in new_nodes:
                        if n.id != self.local_node.id:
                            self.routing_table.add_contact(n)
                            # Add to shortlist if not already there
                            if n not in shortlist:
                                shortlist.append(n)
            
            # Re-sort
            shortlist.sort(key=lambda n: n.id ^ target_id)
            # K-bucket size limit
            shortlist = shortlist[:20]
            
            if not results_found:
                converged = True
            
        return shortlist

    def lookup_value(self, key: int) -> Optional[str]:
        """
        Find a value in the DHT.
        Returns the value if found, None otherwise.
        """
        shortlist = self.routing_table.find_closest(key)
        if not shortlist:
            return None
            
        visited = set()
        visited.add(self.local_node.id)
        
        converged = False
        while not converged:
            candidates = [n for n in shortlist if n.id not in visited][:3]
            if not candidates:
                break
                
            results_found = False
            for node in candidates:
                visited.add(node.id)
                
                val, nodes = self.find_value(node, key)
                if val:
                    return val # Found it!
                    
                if nodes:
                    results_found = True
                    for n in nodes:
                        if n.id != self.local_node.id:
                            self.routing_table.add_contact(n)
                            if n not in shortlist:
                                shortlist.append(n)
                                
            shortlist.sort(key=lambda n: n.id ^ key)
            shortlist = shortlist[:20]
            
            if not results_found:
                converged = True
                
        return None

    def announce(self, key_string: str):
        """
        Announce a string key (e.g. "layer_0") to the DHT.
        The key is hashed to find the location.
        Value is our connection info.
        """
        import hashlib
        import json
        # Hash the key string to get the 160-bit Key ID
        key_id = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        value = f"{self.local_node.ip}:{self.local_node.port}"
        
        # CRITICAL: ALWAYS store locally so other nodes can find us!
        # When we're a solo node, we ARE the closest node to any key.
        # Without local storage, DHT lookups will fail.
        try:
            # Store as list to support multiple holders
            existing = self.storage.get(key_id)
            if existing:
                holders = json.loads(existing)
                if value not in holders:
                    holders.append(value)
            else:
                holders = [value]
            self.storage[key_id] = json.dumps(holders)
        except Exception as e:
            logger.debug(f"Local DHT storage failed: {e}")
        
        # Find K closest nodes to the KEY ID (for replication to other nodes)
        nodes = self.lookup_node(key_id)
        
        store_count = 0
        for node in nodes:
            try:
                if self.store(node, key_id, value):
                    store_count += 1
            except: pass
        
        if store_count > 0:
            # Rate limit success logs to avoid spam (heartbeats happen every 10s)
            now = time.time()
            last_success = getattr(self, '_last_announce_success', {}).get(key_string, 0)
            if not hasattr(self, '_last_announce_success'):
                self._last_announce_success = {}
            
            # Only log first announce and then every 5 minutes
            if last_success == 0 or (now - last_success) > 300:
                logger.info(f"DHT Announce success: Stored '{key_string}' on {store_count} nodes.")
                self._last_announce_success[key_string] = now
        else:
            # Rate limit all DHT announce logs to avoid spam
            now = time.time()
            last_log = self._last_no_peers_log.get(key_string, 0)
            if now - last_log > 60:  # Rate limit: once per minute per key
                if len(nodes) == 0:
                    # No peers found - expected when you're the first/only node
                    logger.debug(f"DHT Announce: No peers found to store '{key_string}' (this is normal when you're the first node).")
                elif len(nodes) <= 3:
                    # Few nodes found and store failed - common in small networks or Docker
                    logger.debug(f"DHT Announce: Store failed for '{key_string}' (found {len(nodes)} nodes). This is normal in small networks.")
                else:
                    # Found many nodes but all failed - this might be a real problem
                    logger.warning(f"DHT Announce failed: Could not store '{key_string}' (found {len(nodes)} nodes but store failed).")
                self._last_no_peers_log[key_string] = now
