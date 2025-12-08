"""
Proof of Neural Work - Semantic Verifier

This module enforces UNIVERSAL verification rules - the "laws of physics" that apply
to ALL nodes regardless of their internal state. These are constraints that no
legitimate node can violate.

Architecture:
============
The verification system has two layers:

1. ProofVerifier (this file) - Universal Constraints
   - Physical rate limits (no GPU can exceed certain throughput)
   - Required fields validation
   - Format and sanity checks
   - Delegates to model_interface for node-specific checks

2. ModelInterface (e.g., SwarmEnabledDynamicNode) - Node-Specific State
   - Internal counter verification (only the node knows its true work count)
   - Model hash matching against local architecture
   - Training enabled status

The Verifier enforces *laws of physics*.
The Node enforces *personal integrity*.
"""

import logging
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS - Hardware Limits
# =============================================================================
# These are generous upper bounds based on current GPU capabilities.
# Even an H100 cluster cannot exceed these sustained rates.

# Maximum training batches per second (sustained)
# Rationale: Even H100 with small batch takes ~50ms per step minimum
# 2.0 batches/sec = 500ms/batch, very generous for any real training
MAX_TRAINING_RATE_PER_SEC = 2.0

# Maximum inference tokens per second per GPU
# Rationale: H100 can do ~3000 tokens/sec for small models, ~500 for large
# 5000 is generous upper bound for any realistic deployment
MAX_INFERENCE_TOKENS_PER_SEC = 5000.0

# Minimum uptime to claim meaningful work (prevents micro-farming)
MIN_UPTIME_FOR_TRAINING_REWARD = 10.0  # 10 seconds


class ProofVerifier:
    """
    Verifies the semantic validity of Proof of Neural Work.
    
    Enforces universal constraints that apply to ALL nodes:
    - Physical hardware limits (rate caps)
    - Required field validation
    - Proof format sanity
    
    Delegates node-specific verification to the model_interface:
    - Internal work counter validation
    - Local model hash matching
    - Training state verification
    
    Usage:
        verifier = ProofVerifier(model_interface=swarm_node)
        is_valid, reason = verifier.verify_work_content(proof)
    """
    
    def __init__(self, model_interface: Optional[Any] = None):
        """
        Initialize the verifier.
        
        Args:
            model_interface: Object implementing verify_training_work(proof).
                             Typically a SwarmEnabledDynamicNode.
                             Required for training proof verification.
        """
        self.model_interface = model_interface
        
    def verify_work_content(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify the content of a PoNW proof.
        
        Applies universal constraints first, then delegates to
        model_interface for node-specific validation.
        
        Args:
            proof: The PoNWProof object to verify
            
        Returns:
            (is_valid, reason) tuple
        """
        proof_type = getattr(proof, 'proof_type', None)
        
        if proof_type == "training":
            return self._verify_training_work(proof)
        elif proof_type == "inference":
            return self._verify_inference_work(proof)
        elif proof_type == "uptime":
            return self._verify_uptime_work(proof)
        elif proof_type == "data":
            return self._verify_data_work(proof)
        else:
            return False, f"Unknown proof type: {proof_type}"
            
    def _verify_training_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify training proof with both universal and node-specific checks.
        
        Universal Checks (this method):
        1. Required fields present (model_hash)
        2. Physical rate limit (can't exceed hardware capabilities)
        3. Minimum uptime threshold
        
        Node-Specific Checks (delegated to model_interface):
        4. Model hash matches local architecture
        5. Claimed batches <= internal counter
        6. Training is actually enabled
        
        The "Golden Rule" of PoNW: L(w - lr * g) < L(w)
        We verify this indirectly through counter checks.
        """
        # =====================================================================
        # UNIVERSAL CHECK 1: Required Fields
        # =====================================================================
        if not getattr(proof, 'model_hash', None):
            return False, "Missing model hash for training proof"
        
        # =====================================================================
        # UNIVERSAL CHECK 2: Physical Rate Limit
        # =====================================================================
        # No GPU can sustain more than MAX_TRAINING_RATE_PER_SEC batches/second.
        # This is a hard physical constraint.
        uptime = getattr(proof, 'uptime_seconds', 0)
        batches = getattr(proof, 'training_batches', 0)
        
        if uptime > 0 and batches > 0:
            rate = batches / uptime
            if rate > MAX_TRAINING_RATE_PER_SEC:
                return False, (
                    f"Impossible training rate: {rate:.2f} batches/sec "
                    f"(max: {MAX_TRAINING_RATE_PER_SEC})"
                )
        
        # =====================================================================
        # UNIVERSAL CHECK 3: Minimum Uptime
        # =====================================================================
        # Prevent micro-farming attacks with tiny uptimes
        if batches > 0 and uptime < MIN_UPTIME_FOR_TRAINING_REWARD:
            return False, (
                f"Uptime too short for training reward: {uptime:.1f}s "
                f"(min: {MIN_UPTIME_FOR_TRAINING_REWARD}s)"
            )
        
        # =====================================================================
        # NODE-SPECIFIC CHECKS: Delegate to ModelInterface
        # =====================================================================
        # These checks require knowledge of the node's internal state.
        if self.model_interface is None:
            return False, "Cannot verify training work: no model interface available"
        
        if not hasattr(self.model_interface, 'verify_training_work'):
            return False, "Model interface missing verify_training_work method"
        
        return self.model_interface.verify_training_work(proof)

    def _verify_inference_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify inference proof via economic receipts.
        
        Inference rewards flow from User -> Node.
        We verify that the User authorized this work.
        
        Universal Checks:
        1. Request ID present (links to payment)
        2. Token rate within physical limits
        """
        # =====================================================================
        # UNIVERSAL CHECK 1: Request ID Required
        # =====================================================================
        if not getattr(proof, 'request_id', None):
            return False, "Missing request_id: anonymous inference not verifiable"
        
        # =====================================================================
        # UNIVERSAL CHECK 2: Token Rate Limit
        # =====================================================================
        uptime = getattr(proof, 'uptime_seconds', 0)
        tokens = getattr(proof, 'tokens_processed', 0)
        
        if uptime > 0 and tokens > 0:
            rate = tokens / uptime
            if rate > MAX_INFERENCE_TOKENS_PER_SEC:
                return False, (
                    f"Impossible inference rate: {rate:.0f} tokens/sec "
                    f"(max: {MAX_INFERENCE_TOKENS_PER_SEC})"
                )
        
        # Inference receipts are further validated by the Market/Ledger
        # which checks that the user actually paid for this request
        return True, "Inference work linked to valid request"
    
    def _verify_uptime_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify uptime proof.
        
        Uptime is verified through:
        1. Gossip protocol (peers attest to availability)
        2. Rate limits in the Ledger (can't claim faster than real time)
        
        The Ledger handles most uptime validation through its
        rate limiting and gossip-based peer attestation.
        """
        uptime = getattr(proof, 'uptime_seconds', 0)
        
        # Sanity check: uptime can't be negative
        if uptime < 0:
            return False, "Negative uptime is invalid"
        
        # Sanity check: uptime can't exceed proof window
        # (The Ledger enforces this more precisely with timestamps)
        max_reasonable_uptime = 3600 * 24  # 24 hours max per proof
        if uptime > max_reasonable_uptime:
            return False, f"Uptime {uptime}s exceeds maximum proof window"
        
        return True, "Uptime valid"
    
    def _verify_data_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify data serving proof.
        
        Data serving is verified optimistically:
        - Nodes claim they served data shards
        - The tracker/DHT can spot-check availability
        - Economic incentives discourage false claims
        
        Future: Could add Merkle proofs for data possession.
        """
        layers_held = getattr(proof, 'layers_held', 0)
        
        # Sanity check: can't hold negative layers
        if layers_held < 0:
            return False, "Negative layers_held is invalid"
        
        # Sanity check: can't hold more layers than exist
        max_layers = 128  # Generous upper bound
        if layers_held > max_layers:
            return False, f"layers_held {layers_held} exceeds maximum {max_layers}"
        
        return True, "Data serving valid (optimistic verification)"
