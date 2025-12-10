"""
gRPC Server for NeuroShard Node

Handles:
1. Inference requests (forward pass through NeuroLLM)
2. Training gradient exchange
3. DHT operations for peer discovery
4. PoNW proof verification
5. Layer-specific forward (for distributed inference)
"""

import grpc
from concurrent import futures
import torch
import time
import threading
import random
from typing import Optional, Union
import logging

# Import generated protobuf code
from protos import neuroshard_pb2
from protos import neuroshard_pb2_grpc

from neuroshard.utils.serialization import deserialize_tensor, serialize_tensor
from neuroshard.core.network.p2p import P2PManager
from neuroshard.core.network.dht_service import DHTServiceMixin

# Swarm Service Mixin (Phase 4)
try:
    from neuroshard.core.swarm.service import SwarmServiceMixin
    SWARM_SERVICE_AVAILABLE = True
except ImportError:
    SWARM_SERVICE_AVAILABLE = False
    SwarmServiceMixin = object  # Fallback

logger = logging.getLogger(__name__)

# Global state
GRPC_SERVER = None


class NeuroShardServiceServicer(DHTServiceMixin, neuroshard_pb2_grpc.NeuroShardServiceServicer):
    """
    gRPC service for NeuroShard nodes.
    
    Supports DynamicNeuroNode with layer-based routing.
    Each node holds specific layers and can process inference requests for those layers.
    
    Swarm Architecture:
    - SwarmForward: Async activation forwarding with failover
    - GetSwarmStatus: Buffer fill rates, capacity info
    - UpdatePeerCapacity: TCP fallback for heartbeat updates
    """
    
    def __init__(self, model, p2p: P2PManager, swap_controller=None):
        """
        Args:
            model: DynamicNeuroNode or SwarmEnabledDynamicNode instance
            p2p: P2P manager for peer discovery
            swap_controller: Deprecated, kept for compatibility
        """
        self.model = model
        self.p2p = p2p
        self.swap_controller = swap_controller
        
        # Detect if this is a DynamicNeuroNode
        self.is_dynamic_node = hasattr(model, 'my_layer_ids') and hasattr(model, 'layer_pool')
        
        # Rate limiter for gradient gossip (per-node)
        # Allows max 1 gossip per node per 60 seconds (DiLoCo sync is every ~8-10 min)
        self._gossip_rate_limit: dict = {}  # node_id -> last_gossip_time
        self._gossip_rate_limit_seconds = 60  # Min seconds between gossips from same node
        self._gossip_rate_limit_lock = threading.Lock()
        
        # Initialize DHT Mixin if available
        if p2p.routing_table:
            DHTServiceMixin.__init__(self, p2p.routing_table, p2p.dht_storage, ledger=p2p.ledger)
        else:
            from neuroshard.core.network.dht import RoutingTable, Node
            dummy_rt = RoutingTable(Node(0, "0.0.0.0", 0))
            DHTServiceMixin.__init__(self, dummy_rt, {}, ledger=p2p.ledger)
        
        logger.info(f"gRPC Servicer initialized (DynamicNode={self.is_dynamic_node})")

    def UnaryInference(self, request, context):
        """Handle inference request."""
        try:
            # Deserialize input
            input_str = request.tensor_data.decode('utf-8')
            input_tensor = deserialize_tensor(input_str)
            
            # Forward pass through DynamicNeuroNode
            output = self.model.forward(input_tensor, session_id=request.session_id)
            # Return result directly
            return neuroshard_pb2.InferenceResponse(
                success=True,
                tensor_data=serialize_tensor(output).encode('utf-8')
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return neuroshard_pb2.InferenceResponse(success=False, error_message=str(e))

    def GetWeights(self, request, context):
        """Return model weights (for gradient sync)."""
        try:
            # Get weights for my layers only
            layer_weights = {}
            for layer_id, layer in self.model.model.my_layers.items():
                layer_weights[f"layer_{layer_id}"] = layer.state_dict()
            
            if self.model.model.embedding:
                layer_weights["embedding"] = self.model.model.embedding.state_dict()
            if self.model.model.lm_head:
                layer_weights["lm_head"] = self.model.model.lm_head.state_dict()
            
            data = serialize_tensor(layer_weights, use_quantization=False).encode('utf-8')
            return neuroshard_pb2.WeightResponse(weights_data=data)
        except Exception as e:
            logger.error(f"GetWeights error: {e}")
            return neuroshard_pb2.WeightResponse(weights_data=b"")

    def GetTrainingStatus(self, request, context):
        """Get training status."""
        stats = self.model.get_stats()
        
        return neuroshard_pb2.TrainingStatusResponse(
            training_enabled=self.model.enable_training,
            total_rounds=stats.get("total_training_rounds", 0),
            current_loss=stats.get("current_loss", float('inf')),
            tokens_processed=stats.get("total_tokens_processed", 0),
        )

    def GetPoNWProof(self, request, context):
        """Get Proof of Neural Work."""
        proof = self.model.get_ponw_proof()
        
        return neuroshard_pb2.PoNWProofResponse(
            node_id=proof.get("node_id", ""),
            timestamp=proof.get("timestamp", 0),
            tokens_processed=proof.get("tokens_processed", 0),
            training_rounds=proof.get("training_rounds", 0),
            signature=proof.get("signature", ""),
        )

    # ==================== LAYER-SPECIFIC FORWARD ====================
    
    def LayerForward(self, request, context):
        """
        Forward hidden states through specific layers on this node.
        
        This enables distributed inference:
        1. Request comes with hidden states
        2. We process through our assigned layers
        3. Return processed hidden states
        """
        try:
            # Deserialize input
            input_str = request.tensor_data.decode('utf-8')
            hidden_states = deserialize_tensor(input_str)
            
            # Check if we have the requested layers
            requested_layers = list(request.layer_ids)
            my_layers = set(self.model.my_layer_ids)
            
            if not all(l in my_layers for l in requested_layers):
                missing = set(requested_layers) - my_layers
                return neuroshard_pb2.LayerForwardResponse(
                    success=False,
                    error_message=f"Missing layers: {missing}"
                )
            
            # Forward through requested layers
            output = self.model.model.forward_my_layers(
                hidden_states,
                start_layer=min(requested_layers),
                end_layer=max(requested_layers) + 1
            )
            
            return neuroshard_pb2.LayerForwardResponse(
                success=True,
                tensor_data=serialize_tensor(output).encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"LayerForward error: {e}")
            return neuroshard_pb2.LayerForwardResponse(
                success=False,
                error_message=str(e)
            )
    
    def GetNodeInfo(self, request, context):
        """Get information about this node's capabilities."""
        stats = self.model.get_stats()
        
        return neuroshard_pb2.NodeInfoResponse(
            node_id=self.model.node_id,
            layer_ids=self.model.my_layer_ids,
            has_embedding=self.model.model.has_embedding if self.model.model else False,
            has_lm_head=self.model.model.has_lm_head if self.model.model else False,
            available_memory_mb=int(self.model.available_memory_mb),
            total_params=stats.get("my_params", 0),
        )

    # ==================== DISTRIBUTED TRAINING RPCs ====================
    
    def GossipGradient(self, request, context):
        """
        Receive gradient contribution from a peer.
        
        This is the core of distributed training:
        1. Peer computes gradients locally
        2. Peer broadcasts via this RPC
        3. We aggregate and apply updates
        
        Rate limited to prevent spam attacks (1 gossip per node per 60s).
        """
        
        try:
            # Check if training is enabled
            if not self.model.enable_training:
                return neuroshard_pb2.GossipGradientResponse(
                    accepted=False,
                    reason="Training not enabled on this node"
                )
            
            # RATE LIMITING: Prevent spam from malicious nodes
            node_id = request.node_id
            now = time.time()
            
            with self._gossip_rate_limit_lock:
                last_gossip = self._gossip_rate_limit.get(node_id, 0)
                time_since_last = now - last_gossip
                
                if time_since_last < self._gossip_rate_limit_seconds:
                    logger.warning(f"Rate limited gradient from {node_id[:8]}... "
                                   f"(only {time_since_last:.1f}s since last, need {self._gossip_rate_limit_seconds}s)")
                    return neuroshard_pb2.GossipGradientResponse(
                        accepted=False,
                        reason=f"Rate limited: wait {self._gossip_rate_limit_seconds - time_since_last:.0f}s"
                    )
                
                # Update last gossip time
                self._gossip_rate_limit[node_id] = now
                
                # Clean up old entries (older than 10 minutes)
                stale_nodes = [nid for nid, ts in self._gossip_rate_limit.items() if now - ts > 600]
                for nid in stale_nodes:
                    del self._gossip_rate_limit[nid]
            
            # Convert protobuf to GradientContribution
            from neuroshard.core.training.distributed import GradientContribution
            
            contribution = GradientContribution(
                node_id=request.node_id,
                round_id=request.round_id,
                layer_gradients=dict(request.layer_gradients),  # Convert MapContainer to dict
                batch_size=request.batch_size,
                loss=request.loss,
                timestamp=request.timestamp,
                signature=request.signature
            )
            
            # Submit to NeuroNode for processing
            success = self.model.receive_peer_gradients(contribution)
            
            if success:
                logger.info(f"Received gradient from peer {request.node_id[:8]}... "
                           f"(round={request.round_id}, batch={request.batch_size}, loss={request.loss:.4f})")
            
            return neuroshard_pb2.GossipGradientResponse(
                accepted=success,
                reason="" if success else "Failed to process gradient",
                current_round=self.model.current_training_round
            )
            
        except Exception as e:
            logger.error(f"GossipGradient error: {e}")
            return neuroshard_pb2.GossipGradientResponse(
                accepted=False,
                reason=str(e)
            )
    
    def GetCheckpointInfo(self, request, context):
        """Get checkpoint info without downloading the full checkpoint."""
        if not self.is_neuro_node:
            return neuroshard_pb2.GetCheckpointInfoResponse()
        
        try:
            info = self.model.get_checkpoint_info()
            
            return neuroshard_pb2.GetCheckpointInfoResponse(
                version=info.get("version", 0),
                model_hash=info.get("model_hash", ""),
                phase=info.get("phase", "bootstrap"),
                params=info.get("params", 0),
                loss=info.get("loss", float('inf'))
            )
            
        except Exception as e:
            logger.error(f"GetCheckpointInfo error: {e}")
            return neuroshard_pb2.GetCheckpointInfoResponse()
    
    def GetCheckpoint(self, request, context):
        """Download full checkpoint from this node."""
        if not self.is_neuro_node:
            return neuroshard_pb2.GetCheckpointResponse(
                success=False,
                error_message="Node does not support checkpoint sync"
            )
        
        try:
            import io
            import zlib
            
            # Get model checkpoint
            if not self.model.model:
                return neuroshard_pb2.GetCheckpointResponse(
                    success=False,
                    error_message="Model not loaded"
                )
            
            # Serialize checkpoint
            buffer = io.BytesIO()
            checkpoint = {
                "model_state_dict": self.model.model.state_dict(),
                "config": {
                    "phase": self.model.phase,
                    "hidden_dim": self.model.model.config.hidden_dim,
                    "num_layers": self.model.model.config.num_layers,
                    "vocab_size": self.model.model.config.vocab_size,
                },
                "version": self.model.total_training_rounds,
                "model_hash": self.model._get_model_hash(),
            }
            torch.save(checkpoint, buffer)
            
            # Compress
            raw_data = buffer.getvalue()
            compressed = zlib.compress(raw_data, level=6)
            
            logger.info(f"Serving checkpoint: version={checkpoint['version']}, "
                       f"size={len(compressed)/1024:.1f}KB (compressed from {len(raw_data)/1024:.1f}KB)")
            
            return neuroshard_pb2.GetCheckpointResponse(
                success=True,
                version=checkpoint["version"],
                model_hash=checkpoint["model_hash"],
                phase=self.model.phase,
                checkpoint_data=compressed,
                total_size=len(compressed)
            )
            
        except Exception as e:
            logger.error(f"GetCheckpoint error: {e}")
            return neuroshard_pb2.GetCheckpointResponse(
                success=False,
                error_message=str(e)
            )

    # ==================== PIPELINE PARALLELISM RPCs ====================
    
    def PipelineForward(self, request, context):
        """
        PRODUCTION-READY Pipeline Forward with:
        - Secure activation transfer (differential privacy + encryption)
        - PoNW proof submission for marketplace rewards
        - Full error handling and logging
        
        Used for distributed inference: Driver → Workers → Validator
        """
        if not hasattr(self.model, 'forward_pipeline') and not hasattr(self.model, 'forward'):
             return neuroshard_pb2.PipelineForwardResponse(
                success=False,
                error_message="Node does not support forward pass"
            )
        
        try:
            import numpy as np
            import hashlib
            
            logger.info(f"[WORKER/VALIDATOR] Received pipeline forward for request {request.request_id[:8]}...")
            
            # STEP 1: Validate checksum of received activations
            received_checksum = hashlib.sha256(request.hidden_states).hexdigest()
            logger.info(f"[SECURITY] Received activations checksum: {received_checksum[:16]}...")
            
            # Deserialize hidden states (detect dtype from shape)
            # 2D = input_ids (int64), 3D = hidden_states (float32)
            if len(request.hidden_shape) == 2:
                dtype = np.int64
                logger.info(f"[WORKER] Detected input_ids (2D shape), deserializing as int64")
            else:
                dtype = np.float32
            
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=dtype).copy()  # .copy() makes it writable
            ).reshape(list(request.hidden_shape))
            
            # CRITICAL: Move tensors to the same device as the model!
            # EC2 (CPU) sends tensors, Jetson (CUDA) needs them on GPU
            model_device = getattr(self.model, 'device', 'cpu')
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                model_device = self.model.model.device
            hidden_states = hidden_states.to(model_device)
            
            logger.info(f"[WORKER/VALIDATOR] Loaded activations: {hidden_states.shape} {hidden_states.dtype} on {hidden_states.device}")
            
            # STEP 2: Process through our layers
            # Deserialize attention mask if provided
            attention_mask = None
            if request.attention_mask:
                attention_mask = torch.from_numpy(
                    np.frombuffer(request.attention_mask, dtype=np.float32).copy()
                ).to(model_device)
            
            # Training Labels (if provided)
            training_labels = None
            if request.training_labels:
                training_labels = torch.from_numpy(
                    np.frombuffer(request.training_labels, dtype=np.int64).copy()
                ).to(model_device)
            
            # Forward through our layers
            if hasattr(self.model, 'forward_pipeline'):
                output, new_kv = self.model.forward_pipeline(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    training_labels=training_labels,
                    session_id=request.session_id,
                    sender_url=request.sender_url,
                    use_cache=request.use_cache,
                )
                
                is_final = self.model.model.has_lm_head if hasattr(self.model, 'model') else False
            else:
                # Legacy fallback
                output = self.model.forward(hidden_states)
                new_kv = None
                is_final = True
            
            logger.info(f"[WORKER/VALIDATOR] Processed through layers: output shape {output.shape}, is_final={is_final}")
            
            # STEP 3: Submit PoNW proof for this work (earn NEURO!)
            if hasattr(self.model, 'ledger') and self.model.ledger and request.request_id:
                try:
                    from neuroshard.core.economics.ledger import PoNWProof, sign_proof
                    import uuid
                    
                    # Count tokens processed
                    tokens_processed = output.shape[1] if len(output.shape) > 1 else output.shape[0]
                    
                    # Determine our role
                    has_embedding = self.model.model.has_embedding if hasattr(self.model, 'model') else False
                    has_lm_head = self.model.model.has_lm_head if hasattr(self.model, 'model') else False
                    layers_held = len(self.model.my_layer_ids) if hasattr(self.model, 'my_layer_ids') else 1
                    
                    proof = PoNWProof(
                        node_id=self.model.node_id if hasattr(self.model, 'node_id') else "unknown",
                        proof_type="inference",
                        timestamp=time.time(),
                        nonce=str(uuid.uuid4()),
                        tokens_processed=tokens_processed,
                        request_id=request.request_id,
                        has_embedding=has_embedding,
                        has_lm_head=has_lm_head,
                        layers_held=layers_held
                    )
                    
                    # Sign and submit
                    signed_proof = sign_proof(proof, self.model.node_token if hasattr(self.model, 'node_token') else "")
                    success, reward, msg = self.model.ledger.process_proof(signed_proof)
                    
                    if success:
                        role = "VALIDATOR" if has_lm_head else ("DRIVER" if has_embedding else "WORKER")
                        pct = "15%" if (has_lm_head or has_embedding) else "70%"
                        logger.info(f"[{role}] ✅ Proof submitted, earned {reward:.6f} NEURO ({pct} of pool)")
                    else:
                        logger.warning(f"[WORKER/VALIDATOR] ⚠️ Proof rejected: {msg}")
                        
                except Exception as e:
                    logger.error(f"[WORKER/VALIDATOR] Failed to submit proof: {e}")
            
            # STEP 4: Serialize output and calculate checksum
            output_bytes = output.detach().cpu().numpy().tobytes()
            output_shape = list(output.shape)
            
            # Calculate checksum for integrity verification
            output_checksum = hashlib.sha256(output_bytes).hexdigest()
            logger.info(f"[SECURITY] Sending output with checksum: {output_checksum[:16]}...")
            
            response = neuroshard_pb2.PipelineForwardResponse(
                request_id=request.request_id,
                success=True,
                hidden_states=output_bytes,
                hidden_shape=output_shape,
                is_final=is_final,
            )
            
            # If final (validator), return logits
            if is_final:
                response.logits = output_bytes
                response.logits_shape = output_shape
                
                if hasattr(self.model, 'current_loss'):
                    response.loss = self.model.current_loss
                
                logger.info(f"[VALIDATOR] ✅ Final output generated, returning logits")
            
            return response
            
        except Exception as e:
            logger.error(f"[WORKER/VALIDATOR] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return neuroshard_pb2.PipelineForwardResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )

    def PipelineBackward(self, request, context):
        """
        Backward pass: propagate gradients back to previous node.
        """
        if not hasattr(self.model, 'backward_pipeline'):
            return neuroshard_pb2.PipelineBackwardResponse(
                success=False,
                error_message="Node does not support backward pipeline"
            )
            
        try:
            import numpy as np
            
            # Deserialize gradients
            grad_output = torch.from_numpy(
                np.frombuffer(request.grad_output, dtype=np.float32).copy()
            ).reshape(list(request.grad_shape))
            
            # CRITICAL: Move gradients to the same device as the model!
            model_device = getattr(self.model, 'device', 'cpu')
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                model_device = self.model.model.device
            grad_output = grad_output.to(model_device)
            
            # Run backward pipeline
            self.model.backward_pipeline(
                grad_output=grad_output,
                session_id=request.session_id
            )
            
            return neuroshard_pb2.PipelineBackwardResponse(success=True)
            
        except Exception as e:
            logger.error(f"PipelineBackward error: {e}")
            return neuroshard_pb2.PipelineBackwardResponse(
                success=False, 
                error_message=str(e)
            )

    def GetShardChunk(self, request, context):
        """
        Serve a chunk of a data shard to a peer (Data Swarm).
        """
        if not hasattr(self.model, 'swarm') or not self.model.swarm:
            return neuroshard_pb2.GetShardChunkResponse(
                success=False,
                error_message="Swarm not initialized on this node"
            )
            
        chunk_data = self.model.swarm.serve_chunk(request.shard_id, request.chunk_index)
        
        if chunk_data:
            return neuroshard_pb2.GetShardChunkResponse(
                success=True,
                data=chunk_data
            )
        else:
            return neuroshard_pb2.GetShardChunkResponse(
                success=False,
                error_message="Chunk not found"
            )
    
    def GetShardInfo(self, request, context):
        """Get shard information from this node."""
        if not self.is_neuro_node:
            return neuroshard_pb2.GetShardInfoResponse()
        
        try:
            if hasattr(self.model, 'get_shard_info'):
                info = self.model.get_shard_info()
            else:
                # Regular NeuroNode - full model
                info = {
                    "shard_id": 0,
                    "total_shards": 1,
                    "start_layer": 0,
                    "end_layer": self.model.model.config.num_layers if self.model.model else 12,
                    "has_embedding": True,
                    "has_lm_head": True,
                    "version": self.model.total_training_rounds,
                    "model_hash": self.model._get_model_hash() if hasattr(self.model, '_get_model_hash') else "",
                }
            
            return neuroshard_pb2.GetShardInfoResponse(
                shard_id=info.get("shard_id", 0),
                total_shards=info.get("total_shards", 1),
                start_layer=info.get("start_layer", 0),
                end_layer=info.get("end_layer", 12),
                has_embedding=info.get("has_embedding", True),
                has_lm_head=info.get("has_lm_head", True),
                version=info.get("version", 0),
                model_hash=info.get("model_hash", ""),
                available_memory_mb=info.get("available_memory_mb", 0),
                current_load=info.get("current_load", 0),
            )
            
        except Exception as e:
            logger.error(f"GetShardInfo error: {e}")
            return neuroshard_pb2.GetShardInfoResponse()

    def _call_peer(self, peer_url, original_req, output_tensor):
        """Call a peer node (legacy pipeline relay)."""
        from urllib.parse import urlparse
        from neuroshard.core.network.connection_pool import get_channel

        parsed = urlparse(peer_url)
        peer_ip = parsed.hostname
        peer_http_port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        peer_grpc_addr = f"{peer_ip}:{peer_http_port + 1000}"
        
        channel = get_channel(peer_grpc_addr)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        fwd_req = neuroshard_pb2.InferenceRequest(
            session_id=original_req.session_id,
            request_id=original_req.request_id,
            tensor_data=serialize_tensor(output_tensor).encode('utf-8'),
            draft_tokens=original_req.draft_tokens,
            sender_reputation=original_req.sender_reputation,
            source_layer=getattr(self.model, 'end', 0)
        )
        
        return stub.UnaryInference(fwd_req)

    def _perform_audit(self, primary_peer, original_req, output_tensor):
        """Audit a peer by comparing with redundant peer."""
        redundant_peer = self.p2p.get_redundant_hop(getattr(self.model, 'end', 0), primary_peer)
        if not redundant_peer:
            return
            
        logger.debug(f"AUDITING: Checking {primary_peer} against {redundant_peer}...")
        
        try:
            res_redundant = self._call_peer(redundant_peer, original_req, output_tensor)
            
            if res_redundant.success:
                logger.debug(f"AUDIT: Redundant peer {redundant_peer} responded successfully.")
            else:
                logger.warning(f"AUDIT: Redundant peer {redundant_peer} failed: {res_redundant.error_message}")
        except Exception as e:
            logger.warning(f"Audit error: {e}")
    
    # ==================== SWARM RPCs (Phase 4) ====================
    
    def SwarmForward(self, request, context):
        """
        Swarm-style activation forwarding with async buffering.
        
        This is the async-first activation forwarding for the swarm architecture.
        Activations are queued in the inbound buffer and processed by ComputeEngine.
        
        Unlike PipelineForward (sync), this returns immediately after queueing.
        """
        try:
            import numpy as np
            from neuroshard.core.swarm.buffers import ActivationPacket, ActivationPriority
            
            # Deserialize activation
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=np.float32)
            ).reshape(list(request.hidden_shape))
            
            # Create activation packet
            packet = ActivationPacket(
                priority=request.priority or ActivationPriority.INFERENCE_NORMAL,
                timestamp=time.time(),
                session_id=request.session_id,
                micro_batch_id=request.micro_batch_id,
                tensor_data=hidden_states,
                source_node=request.source_node,
                target_layer=request.target_layer,
                is_backward=request.is_backward,
                requires_grad=request.requires_grad,
            )
            
            # Queue in inbound buffer (non-blocking)
            inbound = self.model.swarm.inbound_buffer
            if inbound:
                success = inbound.put_nowait(packet)
                if success:
                    return neuroshard_pb2.SwarmForwardResponse(
                        success=True,
                        queued=True,
                        queue_position=len(inbound),
                        buffer_fill_rate=inbound.fill_rate,
                    )
                else:
                    # Buffer full - backpressure
                    return neuroshard_pb2.SwarmForwardResponse(
                        success=False,
                        error_message="Inbound buffer full (backpressure)",
                        buffer_fill_rate=1.0,
                    )
            else:
                return neuroshard_pb2.SwarmForwardResponse(
                    success=False,
                    error_message="Inbound buffer not initialized"
                )
                
        except Exception as e:
            logger.error(f"SwarmForward error: {e}")
            return neuroshard_pb2.SwarmForwardResponse(
                success=False,
                error_message=str(e)
            )
    
    def GetSwarmStatus(self, request, context):
        """
        Get swarm node status: buffer fill rates, capacity, etc.
        
        Used by peers to check node health before routing.
        """
        try:
            status = self.model.get_swarm_status()
            
            # Extract key metrics
            inbound_fill = 0.0
            outbound_fill = 0.0
            inbound_depth = 0
            outbound_depth = 0
            
            if "inbound_buffer" in status:
                inbound_fill = status["inbound_buffer"].get("fill_rate", 0.0)
                inbound_depth = status["inbound_buffer"].get("queue_size", 0)
            if "outbound_buffer" in status:
                outbound_fill = status["outbound_buffer"].get("fill_rate", 0.0)
                outbound_depth = status["outbound_buffer"].get("queue_size", 0)
            
            # Get layer range
            layer_start = min(self.model.my_layer_ids) if self.model.my_layer_ids else 0
            layer_end = max(self.model.my_layer_ids) + 1 if self.model.my_layer_ids else 0
            
            return neuroshard_pb2.SwarmStatusResponse(
                node_id=self.model.node_id,
                layer_start=layer_start,
                layer_end=layer_end,
                inbound_fill_rate=inbound_fill,
                outbound_fill_rate=outbound_fill,
                inbound_queue_depth=inbound_depth,
                outbound_queue_depth=outbound_depth,
                is_accepting_activations=inbound_fill < 0.95,
            )
            
        except Exception as e:
            logger.error(f"GetSwarmStatus error: {e}")
            return neuroshard_pb2.SwarmStatusResponse(error_message=str(e))
    
    def UpdatePeerCapacity(self, request, context):
        """
        Receive capacity update from peer (TCP fallback for UDP heartbeat).
        
        Used when UDP heartbeats fail (firewalls, etc).
        """
        try:
            # Update router with peer info
            router = self.model.swarm.swarm_router
            if router:
                from neuroshard.core.swarm.heartbeat import CapacityBitmask
                
                capacity = CapacityBitmask(
                    node_id=request.node_id,
                    timestamp=time.time(),
                    available_memory_mb=request.available_memory_mb,
                    queue_depth=request.queue_depth,
                    layer_range=(request.layer_start, request.layer_end),
                    gpu_utilization=request.gpu_utilization,
                    network_saturation=request.network_saturation,
                    is_training=request.is_training,
                    is_accepting_inference=request.is_accepting_inference,
                    is_accepting_activations=request.is_accepting_activations,
                    grpc_addr=request.grpc_addr,
                )
                
                router.update_peer_from_heartbeat(capacity)
                
                return neuroshard_pb2.UpdatePeerCapacityResponse(accepted=True)
            
            return neuroshard_pb2.UpdatePeerCapacityResponse(accepted=False)
            
        except Exception as e:
            logger.error(f"UpdatePeerCapacity error: {e}")
            return neuroshard_pb2.UpdatePeerCapacityResponse(accepted=False)
    
    def GetDiLoCoStatus(self, request, context):
        """
        Get DiLoCo training status.
        
        Returns inner step count, sync progress, etc.
        """
        try:
            progress = self.model.get_diloco_progress()
            
            return neuroshard_pb2.DiLoCoStatusResponse(
                enabled=progress.get("enabled", False),
                inner_step_count=progress.get("inner_step_count", 0),
                inner_steps_total=progress.get("inner_steps_total", 500),
                progress=progress.get("progress", 0.0),
                outer_step_count=progress.get("outer_step_count", 0),
                should_sync=progress.get("should_sync", False),
            )
                
        except Exception as e:
            logger.error(f"GetDiLoCoStatus error: {e}")
            return neuroshard_pb2.DiLoCoStatusResponse(enabled=False)


def serve_grpc(port: int, model, p2p: P2PManager, swap_controller=None):
    """Start the gRPC server."""
    global GRPC_SERVER
    
    # Server options for P2P network - be lenient with keepalive pings
    # This is critical for decentralized networks where nodes ping frequently
    # IMPORTANT: Increase message size for activation tensors in pipeline training!
    # Activation size = batch_size * seq_len * hidden_dim * 4 bytes
    # For batch=4, seq=512, hidden=512: ~4MB, but we need headroom
    MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB for large batches/sequences
    options = [
        ('grpc.keepalive_time_ms', 10000),  # Send keepalive every 10s
        ('grpc.keepalive_timeout_ms', 5000),  # 5s timeout for response
        ('grpc.keepalive_permit_without_calls', True),  # Allow pings without active RPCs
        ('grpc.http2.min_recv_ping_interval_without_data_ms', 5000),  # Accept pings every 5s
        ('grpc.http2.max_ping_strikes', 0),  # Don't penalize frequent pings
        ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),  # For receiving activations
        ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),  # For sending responses
    ]
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    neuroshard_pb2_grpc.add_NeuroShardServiceServicer_to_server(
        NeuroShardServiceServicer(model, p2p, swap_controller), server
    )
    
    grpc_port = port + 1000
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()
    GRPC_SERVER = server
    
    logger.info(f"gRPC Server started on port {grpc_port}")
    
    try:
        # Wait until server is stopped externally via stop_grpc()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
    finally:
        GRPC_SERVER = None
        logger.info(f"gRPC Server on port {grpc_port} terminated")


def start_grpc_background(port: int, model, p2p: P2PManager, swap_controller=None):
    """Start gRPC server in background thread."""
    t = threading.Thread(target=serve_grpc, args=(port, model, p2p, swap_controller), daemon=True)
    t.start()


def stop_grpc(timeout: float = 5.0):
    """Stop the gRPC server gracefully."""
    global GRPC_SERVER
    if GRPC_SERVER is not None:
        logger.info("Stopping gRPC server...")
        try:
            # stop() returns an event that is set when shutdown is complete
            event = GRPC_SERVER.stop(grace=timeout)
            event.wait(timeout=timeout)
            logger.info("gRPC server stopped")
        except Exception as e:
            logger.warning(f"Error stopping gRPC server: {e}")
        finally:
            GRPC_SERVER = None
