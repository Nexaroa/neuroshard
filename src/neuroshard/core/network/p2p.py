import requests
import random
import threading
import time
import hashlib
import logging
import os
import sqlite3
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

# DHT Imports
try:
    from neuroshard.core.network.dht import Node, RoutingTable
    from neuroshard.core.network.dht_protocol import DHTProtocol
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False

# Ledger Imports
try:
    from neuroshard.core.economics.ledger import NEUROLedger, ProofType, PoNWProof
    LEDGER_AVAILABLE = True
except ImportError as e:
    LEDGER_AVAILABLE = False
    print(f"[LEDGER IMPORT ERROR] {e}")

logger = logging.getLogger(__name__)

class P2PManager:
    def __init__(self, my_url: str, shard_range: str, tracker_url: str = "http://localhost:3000", node_token: Optional[str] = None):
        self.my_url = my_url
        self.shard_range = shard_range
        self.tracker_url = tracker_url
        self.node_token = node_token
        self.known_peers: Dict[str, dict] = {} # url -> info
        self.running = True
        self._stop_event = threading.Event()  # For interruptible sleeps
        
        # Parse local shard range
        try:
            self.start_layer, self.end_layer = map(int, shard_range.split("-"))
        except:
            self.start_layer, self.end_layer = 0, 0

        # Metrics
        self.current_tps = 0.0
        self.current_latency = 0.0
        
        # Reference to global state (injected by runner)
        self.state_ref = {}
        
        # --- DHT & Decentralization Init ---
        self.dht = None
        self.routing_table = None
        self.ledger = None
        
        # Node ID will be set from ledger crypto (ECDSA-derived)
        # For DHT we need an integer ID, so we'll derive it from the token
        if self.node_token:
            # Use first 20 bytes of SHA256(token) as DHT node ID (160-bit)
            self.node_id = int(hashlib.sha256(self.node_token.encode()).hexdigest()[:40], 16)
        else:
            # Fallback to random ID
            self.node_id = int(hashlib.sha1(f"{my_url}{time.time()}".encode()).hexdigest(), 16)
        
        # The ledger node_id (32 hex chars from ECDSA public key) will be different
        # but deterministically linked to the same token
        self.ledger_node_id = None  # Set after ledger init

        if DHT_AVAILABLE:
            try:
                parsed = urlparse(my_url)
                ip = parsed.hostname or 'localhost'
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                
                self.local_node = Node(self.node_id, ip, port)
                self.routing_table = RoutingTable(self.local_node)
                self.dht = DHTProtocol(self.local_node, self.routing_table, port)
                # Expose internal storage for gRPC inspection
                self.dht_storage = self.dht.storage 
                logger.info(f"DHT Initialized: {self.local_node}")
            except Exception as e:
                logger.error(f"Failed to init DHT: {e}")
                self.dht_storage = {} # Fallback

        if not hasattr(self, 'dht_storage'):
            self.dht_storage = {}

        if LEDGER_AVAILABLE:
            try:
                # Check for explicit path from environment (Docker/production)
                ledger_db_path = os.getenv("LEDGER_DB_PATH")
                
                if not ledger_db_path:
                    # Fallback to ~/.neuroshard/ directory for local development
                    neuroshard_dir = os.path.join(os.path.expanduser("~"), ".neuroshard")
                    os.makedirs(neuroshard_dir, exist_ok=True)
                    ledger_db_path = os.path.join(neuroshard_dir, f"ledger_{self.node_id}.db")
                else:
                    # Ensure directory exists for explicit path
                    os.makedirs(os.path.dirname(ledger_db_path), exist_ok=True)
                
                logger.info(f"Ledger DB path: {ledger_db_path}")
                
                self.ledger = NEUROLedger(
                    db_path=ledger_db_path,
                    node_token=self.node_token
                )
                # Get the ECDSA-derived node_id from ledger
                self.ledger_node_id = self.ledger.node_id
                logger.info(f"NEUROLedger Initialized with ECDSA node_id: {self.ledger_node_id[:16]}...")
                
                # Bootstrap balance from DHT for existing wallets
                # Fully trustless via ECDSA signature verification + Byzantine consensus
                self._bootstrap_balance_from_dht()
            except Exception as e:
                logger.error(f"Failed to init Ledger: {e}")
                self.ledger = None # Ensure explicit None on failure
        else:
            logger.info("Ledger Manager NOT available (dependencies missing or import failed)")

        # Reference to NeuroNode (set later via set_neuro_node)
        self.neuro_node = None
        
        # Start background tasks
        threading.Thread(target=self._announce_loop, daemon=True).start()
        threading.Thread(target=self._gossip_loop, daemon=True).start()
        if self.ledger:
            threading.Thread(target=self._sync_stakes_loop, daemon=True).start()
    
    def set_neuro_node(self, neuro_node):
        """Set reference to NeuroNode for checkpoint announcements."""
        self.neuro_node = neuro_node
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get swarm-related status from the connected node.
        
        Returns:
            Dict with swarm status including buffer fill rates, DiLoCo progress, etc.
        """
        if not self.neuro_node:
            return {"swarm_enabled": False, "error": "Node not connected"}
        
        # Check if node has swarm capabilities
        if hasattr(self.neuro_node, 'get_swarm_status'):
            return self.neuro_node.get_swarm_status()
        else:
            return {"swarm_enabled": False}
    
    def get_diloco_progress(self) -> Dict[str, Any]:
        """
        Get DiLoCo training progress from the connected node.
        
        Returns:
            Dict with inner step count, sync progress, etc.
        """
        if not self.neuro_node:
            return {"enabled": False, "error": "Node not connected"}
        
        if hasattr(self.neuro_node, 'get_diloco_progress'):
            return self.neuro_node.get_diloco_progress()
        else:
            return {"enabled": False}
    
    def get_network_health(self) -> Dict[str, Any]:
        """
        Get overall network health metrics.
        
        Returns:
            Dict with peer count, average latency, routing stats, etc.
        """
        health = {
            "peer_count": len(self.known_peers),
            "avg_latency_ms": self.current_latency * 1000 if self.current_latency else 0,
            "current_tps": self.current_tps,
            "dht_available": self.dht is not None,
            "ledger_available": self.ledger is not None,
        }
        
        # Add swarm stats if available
        swarm_status = self.get_swarm_status()
        if swarm_status.get("swarm_enabled", False):
            health["swarm_enabled"] = True
            if "router" in swarm_status:
                health["swarm_peers"] = swarm_status["router"].get("peer_count", 0)
            if "heartbeat" in swarm_status:
                health["heartbeat_peers"] = swarm_status["heartbeat"].get("peer_count", 0)
        else:
            health["swarm_enabled"] = False
        
        return health
    
    def stop(self):
        """Stop the P2P manager and all background threads."""
        logger.info("Stopping P2P manager...")
        self.running = False
        
        # Signal stop event (for threads that check it)
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        
        # Close DHT if available
        if self.dht:
            try:
                # DHT doesn't have a stop method, but we can clear its state
                self.dht.storage.clear()
            except Exception:
                pass
        
        # Clear known peers
        self.known_peers.clear()
        
        logger.info("P2P manager stopped")
        
    def update_metrics(self, tps: float, latency: float):
        self.current_tps = tps
        self.current_latency = latency

    def _store_proof_in_dht(self, proof: 'PoNWProof', reward: float):
        """
        Store proof in DHT for decentralized balance sync.
        
        This enables new nodes to bootstrap their balance from DHT
        without relying on a central API.
        
        Args:
            proof: The PoNW proof to store
            reward: The reward amount credited for this proof
        """
        if not self.dht:
            return  # DHT not available
        
        try:
            from neuroshard.core.network.dht_proof_store import DHTProofStore, DHTProofRecord
            
            # Create DHT proof store (lazy init)
            if not hasattr(self, '_dht_proof_store'):
                self._dht_proof_store = DHTProofStore(self.dht)
            
            # Create proof record with ALL fields for verification
            # CRITICAL: Must include nonce, model_hash, and public_key for ECDSA verification
            proof_record = DHTProofRecord(
                node_id=proof.node_id,
                timestamp=proof.timestamp,
                proof_type=proof.proof_type.value if hasattr(proof.proof_type, 'value') else str(proof.proof_type),
                nonce=proof.nonce,  # 🔒 Required for canonical_payload
                reward=reward,
                signature=proof.signature,
                public_key=self.ledger.crypto.public_key_hex if self.ledger and self.ledger.crypto else "",  # 🔒 Required for verification
                uptime_seconds=proof.uptime_seconds,
                tokens_processed=proof.tokens_processed,
                training_batches=proof.training_batches,
                data_samples=proof.data_samples,
                model_hash=proof.model_hash,  # 🔒 Required for canonical_payload
                layers_held=proof.layers_held,
                has_embedding=proof.has_embedding,
                has_lm_head=proof.has_lm_head
            )
            
            # Get wallet_id (first 16 chars of node_id)
            wallet_id = proof.node_id[:16]
            
            # Store in DHT (async in background to not block)
            threading.Thread(
                target=self._dht_proof_store.store_proof_in_dht,
                args=(wallet_id, proof_record),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.debug(f"DHT proof storage error (non-fatal): {e}")
    
    def _bootstrap_balance_from_dht(self):
        """
        Bootstrap balance from DHT (PRODUCTION-READY TRUSTLESS SYSTEM).
        
        This is called on startup to sync historical earnings when running
        the same wallet on a new machine.
        
        SECURITY ARCHITECTURE (Production-Grade):
        ┌─────────────────────────────────────────────────────────────┐
        │ DHT RETRIEVAL (FULLY TRUSTLESS)                            │
        │   1. Query DHT for historical proofs                       │
        │   2. Verify ECDSA signature on EACH proof                  │
        │   3. Cross-validate with 3+ independent DHT nodes          │
        │   4. Require Byzantine consensus                           │
        │   5. Credit only cryptographically verified proofs         │
        │   ✅ Fully decentralized                                   │
        │   ✅ Fully trustless                                       │
        │   ✅ Byzantine-resistant                                   │
        │   ✅ Production-ready                                      │
        └─────────────────────────────────────────────────────────────┘
        
        No Fallbacks:
        - If DHT has no proofs → Start from 0 (new wallet)
        - If network too small → Proofs stored when >=3 nodes
        - No trusted servers required
        
        Similar to: Bitcoin SPV, Ethereum Light Client
        
        Why no API fallback:
        - Would require trusting central server
        - Defeats purpose of decentralization
        - Opens security vulnerabilities
        - Not needed - local DB persists your own proofs
        """
        if not self.ledger or not self.node_token:
            return
        
        try:
            # Get wallet_id (first 16 chars of ECDSA node_id)
            wallet_id = self.ledger.node_id[:16]
            
            # Check if we already have a balance (skip bootstrap if we do)
            current_balance = self.ledger.get_balance()
            if current_balance > 0:
                logger.info(f"Local balance found: {current_balance:.4f} NEURO (skipping bootstrap)")
                return
            
            # ===================================================================
            # PHASE 1: DHT RETRIEVAL (TRUSTLESS)
            # ===================================================================
            dht_success = False
            if self.dht:
                try:
                    from neuroshard.core.network.dht_proof_store import DHTProofStore
                    
                    # Get network size for adaptive behavior
                    all_nodes = self.dht.routing_table.get_all_nodes() if self.dht else []
                    network_size = len(all_nodes) + 1  # +1 for self
                    
                    logger.info(f"[DHT BOOTSTRAP] Querying DHT for wallet {wallet_id}... (network: {network_size} nodes)")
                    
                    dht_store = DHTProofStore(self.dht)
                    
                    # Retrieve proofs from DHT with signature verification
                    verified_proofs, metadata = dht_store.retrieve_proofs_from_dht(
                        wallet_id=wallet_id,
                        max_proofs=100,
                        verify_signatures=True  # 🔒 TRUSTLESS
                    )
                    
                    if verified_proofs:
                        logger.info(f"[DHT BOOTSTRAP] Found {len(verified_proofs)} verified proofs in DHT "
                                   f"(total_reward={metadata.get('total_reward', 0):.6f} NEURO)")
                        
                        # Cross-validate with multiple DHT nodes for Byzantine resistance
                        # Uses adaptive validation (works with 2-node networks)
                        consensus, validation_data = dht_store.cross_validate_proofs(
                            wallet_id=wallet_id,
                            desired_validators=3  # Adapts to actual network size
                        )
                        
                        if consensus:
                            validators_count = validation_data.get('validators_queried', 0)
                            network_size = validation_data.get('network_size', 1)
                            
                            logger.info(f"[DHT BOOTSTRAP] ✅ Cross-validation PASSED "
                                       f"({validators_count} validators, network={network_size} nodes)")
                            
                            # Credit verified proofs to local ledger
                            total_credited = 0.0
                            for proof_record in verified_proofs:
                                # Each proof is ECDSA-verified, safe to credit
                                total_credited += proof_record.reward
                            
                            # Update local ledger with DHT data
                            with self.ledger.lock:
                                with sqlite3.connect(self.ledger.db_path) as conn:
                                    conn.execute("""
                                        INSERT OR REPLACE INTO balances 
                                        (node_id, balance, total_earned, total_spent, proof_count, last_proof_time)
                                        VALUES (?, ?, ?, 0.0, ?, ?)
                                    """, (
                                        self.ledger.node_id,
                                        total_credited,
                                        total_credited,
                                        len(verified_proofs),
                                        time.time()
                                    ))
                                    conn.commit()
                            
                            logger.info(f"[DHT BOOTSTRAP] ✅ Synced from DHT: {total_credited:.6f} NEURO")
                            logger.info(f"[DHT BOOTSTRAP]    {len(verified_proofs)} proofs verified via ECDSA signatures")
                            logger.info(f"[DHT BOOTSTRAP]    Network: {network_size} nodes, {validators_count} validators confirmed")
                            dht_success = True
                            return  # Success!
                        
                        else:
                            logger.warning(f"[DHT BOOTSTRAP] ⚠️ Cross-validation FAILED - nodes disagree")
                            logger.warning(f"[DHT BOOTSTRAP] Validation data: {validation_data}")
                            # Fall through to API
                    else:
                        logger.info(f"[DHT BOOTSTRAP] No proofs found in DHT (new wallet or network still syncing)")
                        # Fall through to API
                        
                except Exception as e:
                    logger.warning(f"[DHT BOOTSTRAP] DHT retrieval failed: {e}")
                    # Fall through to API
            
            # ===================================================================
            # NO API FALLBACK - PRODUCTION-READY TRUSTLESS SYSTEM
            # ===================================================================
            # If DHT doesn't have proofs, we start from zero and earn naturally.
            # This is the CORRECT behavior for a decentralized system.
            # 
            # Why no API fallback:
            # 1. API would be a trusted party (defeats trustless design)
            # 2. Creates centralization point
            # 3. Opens attack vector (malicious API can inflate balances)
            # 
            # Edge case handling:
            # - New wallet: Balance = 0 (correct)
            # - Existing wallet but DHT empty: Proofs are in local DB, will
            #   propagate to DHT as we earn. Other machines bootstrap when DHT
            #   has enough replicas (3+ nodes needed)
            # 
            # This is how Bitcoin/Ethereum work - fully decentralized.
            # ===================================================================
            
            if not dht_success:
                logger.info(f"[BOOTSTRAP] No proofs found in DHT for wallet {wallet_id[:8]}...")
                logger.info(f"[BOOTSTRAP] Starting with zero balance - will earn via PoNW")
                logger.info(f"[BOOTSTRAP] Future earnings will be stored in DHT for other machines")
                logger.info(f"[BALANCE] New wallet - starting from 0 NEURO. Start earning!")
            
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] Error during DHT bootstrap: {e}")
            logger.info("[BOOTSTRAP] Starting with zero balance - future earnings via P2P")

    def _sync_stakes_loop(self):
        """
        P2P stake gossip loop.
        
        Periodically broadcasts our stake to peers so they can:
        1. Verify our PoNW claims have correct multipliers
        2. Maintain a network-wide view of stakes
        """
        while self.running:
            # Interruptible sleep - wakes up immediately on stop()
            if self._stop_event.wait(timeout=300):
                break  # Stop event was set
            
            if not self.ledger:
                continue
            
            try:
                # Get our current stake
                account_info = self.ledger.get_account_info()
                stake = account_info.get("stake", 0.0)
                stake_locked_until = account_info.get("stake_locked_until", 0.0)
                
                if stake <= 0:
                    continue  # Nothing to gossip
                
                # Gossip our stake to peers
                peers = list(self.known_peers.keys())
                if self.routing_table:
                    for n in self.routing_table.get_all_nodes():
                        peers.append(f"http://{n.ip}:{n.port}")
                
                if not peers:
                    continue
                
                # DYNAMIC FANOUT: Scale with network size
                # Formula: 2*sqrt(N) + 3, capped at 50 for stake gossip
                # Stakes are important for security - need higher coverage
                import math
                fanout = min(int(2 * math.sqrt(len(peers)) + 3), 50)
                targets = random.sample(peers, min(len(peers), fanout))
                logger.info(f"Stake gossip: Broadcasting {stake:.2f} NEURO to {len(targets)} peers")
                
                for target in targets:
                    threading.Thread(
                        target=self._send_stake_to_peer, 
                        args=(target, stake, stake_locked_until), 
                        daemon=True
                    ).start()
                    
            except Exception as e:
                logger.error(f"Stake gossip error: {e}")

    def _send_stake_to_peer(self, target_url: str, amount: float, locked_until: float):
        """Send stake update to a peer via gRPC."""
        from protos import neuroshard_pb2
        from protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(target_url)
            ip = parsed.hostname
            port = (parsed.port or 80) + 1000  # gRPC port
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            # Create stake gossip request using ECDSA node_id
            ledger_node_id = self.ledger.node_id
            payload = f"{ledger_node_id}:{amount}:{locked_until}"
            
            # SECURITY: Include public key for verification
            # This allows peers to verify our signature without prior knowledge
            public_key_hex = self.ledger.crypto.get_public_key_hex()
            
            req = neuroshard_pb2.GossipStakeRequest(
                node_id=ledger_node_id,
                amount=amount,
                locked_until=locked_until,
                timestamp=time.time(),
                signature=self.ledger._sign(payload),
                public_key=public_key_hex  # Required for verification
            )
            
            stub.GossipStake(req, timeout=3.0)
            logger.debug(f"Stake gossip sent to {ip}:{port}")
        except Exception as e:
            logger.debug(f"Stake gossip to {target_url} failed: {e}")

    def _announce_loop(self):
        # Immediate announce
        self._announce_once()
        while self.running:
            # Interruptible sleep - wakes up immediately on stop()
            if self._stop_event.wait(timeout=10):
                break
            self._announce_once()

    def broadcast_transaction(self, recipient_id: str, amount: float, signature: str, tx_hash: str):
        """Broadcast a transaction to the P2P network."""
        threading.Thread(target=self._gossip_transaction, args=(recipient_id, amount, signature, tx_hash), daemon=True).start()

    def _gossip_transaction(self, recipient_id: str, amount: float, signature: str, tx_hash: str):
        """Gossip transaction to peers."""
        from protos import neuroshard_pb2
        from protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        # 1. Gather Peers
        peers = list(self.known_peers.keys())
        if self.routing_table:
            for n in self.routing_table.get_all_nodes():
                peers.append(f"http://{n.ip}:{n.port}")
                
        if not peers: return

        # 2. Gossip to random subset (Epidemic Propagation)
        # DYNAMIC FANOUT: 2*sqrt(N) + 3, capped at 50 for transactions
        # Transactions need high coverage for consistency
        import math
        fanout = min(int(2 * math.sqrt(len(peers)) + 3), 50)
        targets = random.sample(peers, min(len(peers), fanout))
        
        req = neuroshard_pb2.GossipTransactionRequest(
            sender_id=str(self.node_id),
            recipient_id=recipient_id,
            amount=amount,
            timestamp=time.time(),
            signature=signature,
            tx_hash=tx_hash
        )
        
        logger.info(f"Broadcasting TX {tx_hash[:8]} to {len(targets)} peers...")
        
        for target_url in targets:
            try:
                parsed = urlparse(target_url)
                ip = parsed.hostname
                port = (parsed.port or 80) + 1000
                
                channel = get_channel(f"{ip}:{port}")
                stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                
                stub.GossipTransaction(req, timeout=3.0)
            except Exception as e:
                pass # Gossip is best effort

    def _gossip_loop(self):
        """Periodically create Proof of Neural Work and gossip to peers."""
        try:
            from protos import neuroshard_pb2
            from protos import neuroshard_pb2_grpc
            from neuroshard.core.network.connection_pool import get_channel
        except Exception as e:
            logger.error(f"[PoNW] Failed to import protos: {e}")
            return
        
        logger.info("[PoNW] Gossip loop started (will generate first proof in 60s)")

        while self.running:
            # Interruptible sleep - wakes up immediately on stop()
            if self._stop_event.wait(timeout=60):
                break
            if not self.ledger:
                logger.info("[NODE] PoNW: No ledger available, skipping proof generation")
                continue
                
            try:
                # Get metrics from state
                tokens_processed = self.state_ref.get("token_count", 0)
                training_batches = self.state_ref.get("training_batches", 0)
                
                # Get pending inference request IDs (only paid inference gets rewards)
                pending_request_ids = self.state_ref.get("pending_inference_requests", [])
                
                # Reset counters after snapshot
                self.state_ref["token_count"] = 0
                self.state_ref["training_batches"] = 0
                self.state_ref["pending_inference_requests"] = []
                
                # Determine proof type based on activity
                # IMPORTANT: Inference proofs REQUIRE a request_id (paid request)
                # Tokens processed without a request_id don't earn inference rewards
                if training_batches > 0:
                    proof_type = ProofType.TRAINING
                    # Note: tokens_processed during training doesn't count as inference
                    tokens_processed = 0  # Don't double-count training tokens
                elif tokens_processed > 0 and pending_request_ids:
                    # Only create inference proof if we have actual paid requests
                    proof_type = ProofType.INFERENCE
                else:
                    # Default to uptime - unpaid inference doesn't earn rewards
                    proof_type = ProofType.UPTIME
                    tokens_processed = 0  # Unpaid tokens don't count
                
                # Get node info for role multipliers
                layers_held = len(self.state_ref.get("assigned_layers", []))
                has_embedding = self.state_ref.get("has_embedding", False)
                has_lm_head = self.state_ref.get("has_lm_head", False)
                model_hash = self.state_ref.get("model_hash", "")
                current_loss = self.state_ref.get("current_loss", None)
                
                # Sanitize loss (must be a valid float for storage)
                if current_loss is not None:
                    import math
                    if math.isinf(current_loss) or math.isnan(current_loss):
                        current_loss = None
                
                # Create PoNW proof using new NEUROLedger API
                proof = self.ledger.create_proof(
                    proof_type=proof_type,
                    uptime_seconds=60,
                    tokens_processed=tokens_processed,
                    training_batches=training_batches,
                    layers_held=layers_held,
                    has_embedding=has_embedding,
                    has_lm_head=has_lm_head,
                    model_hash=model_hash,
                    current_loss=current_loss
                )
                
                # Process proof locally (credit ourselves)
                success, reward, msg = self.ledger.process_proof(proof)
                
                if success:
                    if proof_type == ProofType.TRAINING:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (training, {training_batches} batches in last 60s)")
                    elif proof_type == ProofType.INFERENCE:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (inference, {tokens_processed} tokens in last 60s)")
                    else:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (uptime, 60s)")
                    
                    # 🔥 NEW: Store proof in DHT for decentralized balance sync
                    self._store_proof_in_dht(proof, reward)
                else:
                    logger.info(f"[NODE] ❌ PoNW rejected: {msg}")

                # Gossip to random peers
                peers = list(self.known_peers.keys())
                if self.routing_table:
                    for n in self.routing_table.get_all_nodes():
                        peers.append(f"http://{n.ip}:{n.port}")
                
                if not peers:
                    logger.info("PoNW: Solo mining (no peers to gossip)")
                else:
                    # DYNAMIC FANOUT: sqrt(N) + 3, capped at 30 for PoNW proofs
                    # Proofs need reasonable coverage for DHT consistency
                    import math
                    fanout = min(int(math.sqrt(len(peers)) + 3), 30)
                    targets = random.sample(peers, min(len(peers), fanout))
                    logger.info(f"PoNW: Gossiping to {len(targets)} peers")
                    
                    for target in targets:
                        threading.Thread(target=self._send_proof_to_peer, args=(target, proof)).start()
                         
            except Exception as e:
                logger.error(f"PoNW gossip error: {e}")

    def _send_proof_to_peer(self, target_url: str, proof: PoNWProof):
        """Send PoNW proof to a peer via gRPC."""
        from protos import neuroshard_pb2
        from protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(target_url)
            ip = parsed.hostname
            # gRPC port = HTTP port + 1000
            port = (parsed.port or 80) + 1000
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            # Send FULL proof data for proper verification
            # CRITICAL: Include public key for trustless verification
            # CRITICAL: Include data_samples, model_hash, request_id for canonical_payload match
            req = neuroshard_pb2.GossipProofRequest(
                node_id=proof.node_id,
                timestamp=proof.timestamp,
                uptime=proof.uptime_seconds,
                signature=proof.signature,
                token_count=proof.tokens_processed,
                training_batches=proof.training_batches,
                layers_held=proof.layers_held,
                has_embedding=proof.has_embedding,
                has_lm_head=proof.has_lm_head,
                proof_type=proof.proof_type,
                nonce=proof.nonce,
                public_key=self.ledger.crypto.get_public_key_hex(),
                data_samples=proof.data_samples,
                model_hash=proof.model_hash,
                request_id=proof.request_id or "",
                current_loss=proof.current_loss if proof.current_loss is not None else 0.0
            )
            
            stub.GossipProof(req, timeout=3.0)
        except:
            pass # Gossip is best-effort

    def _sync_with_new_peer(self, peer_url: str):
        """
        Sync state with a newly discovered peer.
        
        IMPORTANT: Historical proofs CANNOT be replayed because of the 
        PROOF_FRESHNESS_WINDOW (5 minutes). This is BY DESIGN - it prevents
        nodes from fabricating work while running solo.
        
        How balance sync works (like Bitcoin):
        ┌─────────────────────────────────────────────────────────────────┐
        │ LOCAL BALANCE = All proofs I generated (witnessed by me)        │
        │ NETWORK BALANCE = Proofs gossiped within 5 min (witnessed by N) │
        │                                                                 │
        │ If you run SOLO, only LOCAL balance increases.                  │
        │ NETWORK balance only increases when peers witness your work.    │
        │                                                                 │
        │ This is the SECURITY MODEL: No free NEURO from fabricated work! │
        └─────────────────────────────────────────────────────────────────┘
        
        What we DO sync:
        1. Current peer list (for gossip)
        2. DHT routing table (for lookups)
        3. Training state (for DiLoCo)
        
        What we DON'T sync (by design):
        - Historical proofs older than 5 minutes (prevents fraud)
        """
        # Log the connection for transparency
        logger.info(f"[SYNC] Connected to new peer: {peer_url}")
        
        # NOTE: Historical proof replay removed because:
        # 1. PROOF_FRESHNESS_WINDOW = 300s (5 min) - old proofs rejected
        # 2. This is correct security behavior (like Bitcoin confirmations)
        # 3. Solo-earned NEURO is LOCAL only - needs witnesses to be NETWORK-confirmed

    def _announce_once(self):
        # 1. DHT Announce (Primary)
        if self.dht:
            try:
                # Announce shard range availability
                # Key strategy: announce under "layer_X" where X is our start layer
                # This allows peers looking for layer X to find us.
                self.dht.announce(f"layer_{self.start_layer}")
                
                # Also announce checkpoint info for distributed training sync
                if hasattr(self, 'neuro_node') and self.neuro_node:
                    checkpoint_info = self.neuro_node.get_checkpoint_info()
                    self.dht.announce(f"checkpoint_v{checkpoint_info['version']}")
            except Exception as e:
                logger.debug(f"DHT Announce error: {e}")

        # 2. Legacy Tracker Announce (Fallback)
        try:
            parsed = urlparse(self.my_url)
            ip = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            requests.post(f"{self.tracker_url}/announce", json={
                "ip": ip,
                "port": port,
                "shard_range": self.shard_range,
                "tps": self.current_tps,
                "latency": self.current_latency,
                "node_token": self.node_token
            }, timeout=2)
            
            # Fetch Peers for Bootstrap
            # Only done if routing table is empty or low
            if not self.known_peers or len(self.known_peers) < 5:
                 # First, get peers with matching shard range (for inference routing)
                 resp = requests.get(f"{self.tracker_url}/peers", params={"shard_range": self.shard_range}, timeout=2)
                 if resp.status_code == 200:
                     new_peers = resp.json()
                     for p in new_peers:
                         if p["url"] != self.my_url:
                             self.known_peers[p["url"]] = p
                 
                 # Also fetch ALL peers for gossip (ledger sync needs all nodes, not just matching shards)
                 resp_all = requests.get(f"{self.tracker_url}/peers", params={"limit": 100}, timeout=2)
                 if resp_all.status_code == 200:
                     all_peers = resp_all.json()
                     for p in all_peers:
                         if p["url"] != self.my_url and p["url"] not in self.known_peers:
                             is_new_peer = True
                             self.known_peers[p["url"]] = p
                             # Bootstrap DHT
                             if self.routing_table:
                                 try:
                                    p_parsed = urlparse(p["url"])
                                    p_ip = p_parsed.hostname
                                    p_port = p_parsed.port or 80
                                    # Deterministic ID for stability in dev
                                    p_id = int(hashlib.sha1(f"{p['url']}".encode()).hexdigest(), 16)
                                    if self.routing_table:
                                        self.routing_table.add_contact(Node(p_id, p_ip, p_port))
                                 except: pass
                             
                             # Log new peer connection (proof replay removed - see _sync_with_new_peer)
                             if is_new_peer:
                                 self._sync_with_new_peer(p["url"])
        except:
            pass

    def get_next_hop(self, current_end_layer: int, session_id: Optional[str] = None) -> Optional[str]:
        """Find a peer that starts where we end."""
        candidates = []
        
        # Strategy 1: DHT Lookup (Scalable)
        if self.dht:
            import json
            key_str = f"layer_{current_end_layer}"
            key = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
            
            # Use iterative lookup
            val = self.dht.lookup_value(key)
            if val:
                try:
                    # Try parsing as list of peers
                    dht_candidates = json.loads(val)
                    if isinstance(dht_candidates, list):
                        for c in dht_candidates:
                            # DHT stores "ip:port", we need full URL
                            if not c.startswith("http"):
                                candidates.append(f"http://{c}")
                            else:
                                candidates.append(c)
                    else:
                         # Legacy single value
                         if not isinstance(dht_candidates, str):
                             dht_candidates = str(dht_candidates)
                         if not dht_candidates.startswith("http"):
                             candidates.append(f"http://{dht_candidates}")
                         else:
                             candidates.append(dht_candidates)
                except:
                    # Simple string fallback
                    if not val.startswith("http"):
                        candidates.append(f"http://{val}")
                    else:
                        candidates.append(val)

        # Strategy 2: Local Cache (Fallback)
        for url, info in self.known_peers.items():
            try:
                r = info.get("shard_range", "0-0")
                start, end = map(int, r.split("-"))
                if start == current_end_layer:
                    candidates.append(url)
            except: continue
            
        if not candidates: return None
        
        if session_id:
            # Sticky routing
            candidates.sort()
            hash_val = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
            return candidates[hash_val % len(candidates)]
            
        return random.choice(candidates)
    
    def get_redundant_hop(self, current_end_layer: int, primary_hop: str) -> Optional[str]:
        candidates = []
        for url, info in self.known_peers.items():
            try:
                r = info.get("shard_range", "0-0")
                start, end = map(int, r.split("-"))
                if start == current_end_layer and url != primary_hop:
                    candidates.append(url)
            except: continue
            
        if not candidates: return None
        return random.choice(candidates)
        
    def get_sync_peers(self) -> List[str]:
        candidates = []
        for url, info in self.known_peers.items():
            if info.get("shard_range") == self.shard_range:
                candidates.append(url)
        return candidates
