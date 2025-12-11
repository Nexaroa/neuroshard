"""
Hybrid Validator Consensus System

Implements the stake-weighted proof validation system from the whitepaper (Section 7).

HYBRID VALIDATOR MODEL:
=======================
Validators in NeuroShard combine two roles:
1. MODEL VALIDATION: Compute loss on the last layer (requires LM head)
2. CONSENSUS VALIDATION: Verify PoNW proofs from other nodes (requires stake)

This dual responsibility ensures validators are both:
- Computationally invested (they run the model)
- Economically invested (they stake tokens)

PROOF VALIDATION FLOW:
======================
1. Node generates PoNW proof and gossips to network
2. 3 validators are selected (stake-weighted random)
3. Each validator verifies proof locally and casts vote
4. Votes are gossiped with stake weights
5. If valid_stake / total_stake >= 66%: Accept proof, credit rewards
6. If rejected: No rewards (and slash if fraud detected)
7. Validators who vote against consensus are slashed at 2x rate

VALIDATOR REQUIREMENTS:
=======================
- Minimum 2GB memory (to hold LM head)
- Minimum 100 NEURO staked
- Currently holding the last layer assignment

VALIDATOR REWARDS:
==================
- Model Validation: +30% bonus on all earnings
- Proof Validation: 0.001 NEURO per proof validated
- Stake Multiplier: Up to 1.66x (diminishing returns)
"""

import time
import random
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from enum import Enum

from neuroshard.core.economics.constants import (
    VALIDATOR_MIN_STAKE,
    VALIDATOR_MIN_MEMORY_MB,
    VALIDATION_FEE_PER_PROOF,
    VALIDATION_CONSENSUS_THRESHOLD,
    VALIDATOR_SELECTION_RANDOMNESS,
    VALIDATOR_SLASH_MULTIPLIER,
    SLASH_AMOUNT,
    get_dynamic_validator_stake,
)

logger = logging.getLogger(__name__)


class ConsensusState(Enum):
    """State of a proof in the consensus process."""
    PENDING = "pending"           # Proof submitted, waiting for validators
    COLLECTING_VOTES = "collecting"  # Validators are voting
    CONSENSUS_REACHED = "consensus"  # 66% threshold reached
    ACCEPTED = "accepted"         # Proof accepted, rewards credited
    REJECTED = "rejected"         # Proof rejected by consensus
    TIMEOUT = "timeout"           # Not enough votes in time window


@dataclass
class ValidatorInfo:
    """
    Information about a validator in the network.
    
    A node becomes a validator when it meets ALL requirements:
    1. Holds the last layer (LM head)
    2. Has sufficient stake (100 NEURO, or dynamic based on network size)
    3. Has sufficient memory (>= 2GB)
    """
    node_id: str
    stake: float
    has_lm_head: bool
    memory_mb: int
    url: str  # gRPC URL for validation requests
    last_seen: float = field(default_factory=time.time)
    proofs_validated: int = 0
    correct_votes: int = 0  # Votes that matched consensus
    wrong_votes: int = 0    # Votes against consensus (risk of slashing)
    
    @property
    def is_eligible(self) -> bool:
        """Check if this node is eligible to be a validator."""
        return (
            self.has_lm_head and
            self.stake >= VALIDATOR_MIN_STAKE and
            self.memory_mb >= VALIDATOR_MIN_MEMORY_MB
        )
    
    @property
    def accuracy_rate(self) -> float:
        """Percentage of votes that matched consensus."""
        total = self.correct_votes + self.wrong_votes
        return self.correct_votes / total if total > 0 else 1.0
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "stake": self.stake,
            "has_lm_head": self.has_lm_head,
            "memory_mb": self.memory_mb,
            "url": self.url,
            "is_eligible": self.is_eligible,
            "proofs_validated": self.proofs_validated,
            "accuracy_rate": self.accuracy_rate,
        }


@dataclass
class ValidationVote:
    """A validator's vote on a proof."""
    validator_id: str
    stake: float
    vote: bool  # True = valid, False = invalid
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    signature: str = ""  # ECDSA signature of the vote
    
    def to_dict(self) -> dict:
        return {
            "validator_id": self.validator_id,
            "stake": self.stake,
            "vote": self.vote,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsensusResult:
    """Result of the consensus process for a proof."""
    proof_signature: str
    state: ConsensusState
    total_votes: int = 0
    valid_votes: int = 0
    invalid_votes: int = 0
    valid_stake: float = 0.0
    invalid_stake: float = 0.0
    total_stake: float = 0.0
    consensus_result: Optional[bool] = None  # True=valid, False=invalid, None=no consensus
    validators_slashed: List[str] = field(default_factory=list)
    reward_credited: bool = False
    created_at: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None
    
    @property
    def valid_ratio(self) -> float:
        """Ratio of stake voting valid."""
        return self.valid_stake / self.total_stake if self.total_stake > 0 else 0.0
    
    @property
    def consensus_reached(self) -> bool:
        """Whether 66% threshold was reached."""
        return (
            self.valid_ratio >= VALIDATION_CONSENSUS_THRESHOLD or
            (1 - self.valid_ratio) >= VALIDATION_CONSENSUS_THRESHOLD
        )
    
    def to_dict(self) -> dict:
        return {
            "proof_signature": self.proof_signature,
            "state": self.state.value,
            "total_votes": self.total_votes,
            "valid_votes": self.valid_votes,
            "invalid_votes": self.invalid_votes,
            "valid_stake": self.valid_stake,
            "invalid_stake": self.invalid_stake,
            "valid_ratio": self.valid_ratio,
            "consensus_reached": self.consensus_reached,
            "consensus_result": self.consensus_result,
            "validators_slashed": self.validators_slashed,
            "reward_credited": self.reward_credited,
        }


class ValidatorConsensus:
    """
    Manages the Hybrid Validator Consensus system.
    
    This class coordinates:
    1. Validator registration and tracking
    2. Stake-weighted validator selection
    3. Vote collection and consensus determination
    4. Reward/slash distribution
    
    Usage:
        consensus = ValidatorConsensus(ledger=ledger, p2p=p2p)
        
        # Register as a validator
        consensus.register_validator(node_id, stake, has_lm_head, memory_mb, url)
        
        # Submit proof for consensus
        result = await consensus.submit_proof_for_consensus(proof)
        
        # Cast vote (when selected as validator)
        consensus.cast_vote(proof_signature, vote=True, reason="Valid work")
    """
    
    def __init__(
        self,
        ledger: Any = None,
        p2p: Any = None,
        node_id: str = "",
        num_validators_per_proof: int = 3,
        consensus_timeout_seconds: float = 30.0,
    ):
        """
        Initialize the ValidatorConsensus system.
        
        Args:
            ledger: NEUROLedger for stake verification and reward distribution
            p2p: P2PManager for gossip communication
            node_id: This node's ID
            num_validators_per_proof: Number of validators to select per proof
            consensus_timeout_seconds: Time to wait for votes before timeout
        """
        self.ledger = ledger
        self.p2p = p2p
        self.node_id = node_id
        self.num_validators_per_proof = num_validators_per_proof
        self.consensus_timeout_seconds = consensus_timeout_seconds
        
        # Validator registry: node_id -> ValidatorInfo
        self._validators: Dict[str, ValidatorInfo] = {}
        self._validators_lock = threading.Lock()
        
        # Pending consensus: proof_signature -> ConsensusResult
        self._pending: Dict[str, ConsensusResult] = {}
        self._pending_lock = threading.Lock()
        
        # Votes: proof_signature -> List[ValidationVote]
        self._votes: Dict[str, List[ValidationVote]] = {}
        self._votes_lock = threading.Lock()
        
        # Callbacks
        self._on_consensus_reached: Optional[Callable] = None
        self._verify_proof_locally: Optional[Callable] = None
        
        logger.info(f"ValidatorConsensus initialized: "
                   f"validators_per_proof={num_validators_per_proof}, "
                   f"timeout={consensus_timeout_seconds}s")
    
    # =========================================================================
    # VALIDATOR REGISTRATION
    # =========================================================================
    
    def register_validator(
        self,
        node_id: str,
        stake: float,
        has_lm_head: bool,
        memory_mb: int,
        url: str,
    ) -> bool:
        """
        Register a node as a potential validator.
        
        A node must meet ALL requirements to be eligible:
        1. has_lm_head = True (holds LM head layer)
        2. stake >= VALIDATOR_MIN_STAKE (100 NEURO)
        3. memory_mb >= VALIDATOR_MIN_MEMORY_MB (2GB)
        
        Returns True if registration successful.
        """
        validator = ValidatorInfo(
            node_id=node_id,
            stake=stake,
            has_lm_head=has_lm_head,
            memory_mb=memory_mb,
            url=url,
            last_seen=time.time(),
        )
        
        with self._validators_lock:
            existing = self._validators.get(node_id)
            if existing:
                # Update existing validator
                validator.proofs_validated = existing.proofs_validated
                validator.correct_votes = existing.correct_votes
                validator.wrong_votes = existing.wrong_votes
            
            self._validators[node_id] = validator
        
        if validator.is_eligible:
            logger.info(f"Validator registered: {node_id[:16]}... "
                       f"(stake={stake:.2f}, has_lm_head={has_lm_head})")
        else:
            logger.debug(f"Node registered but not eligible validator: {node_id[:16]}... "
                        f"(stake={stake:.2f}, has_lm_head={has_lm_head}, memory={memory_mb}MB)")
        
        return validator.is_eligible
    
    def unregister_validator(self, node_id: str) -> bool:
        """Remove a validator from the registry."""
        with self._validators_lock:
            if node_id in self._validators:
                del self._validators[node_id]
                logger.info(f"Validator unregistered: {node_id[:16]}...")
                return True
        return False
    
    def update_validator_stake(self, node_id: str, new_stake: float) -> bool:
        """Update a validator's stake amount."""
        with self._validators_lock:
            if node_id in self._validators:
                self._validators[node_id].stake = new_stake
                self._validators[node_id].last_seen = time.time()
                return True
        return False
    
    def get_eligible_validators(self) -> List[ValidatorInfo]:
        """Get all currently eligible validators."""
        with self._validators_lock:
            return [v for v in self._validators.values() if v.is_eligible]
    
    def get_validator_count(self) -> int:
        """Get count of eligible validators."""
        return len(self.get_eligible_validators())
    
    def is_validator(self, node_id: str) -> bool:
        """Check if a node is an eligible validator."""
        with self._validators_lock:
            v = self._validators.get(node_id)
            return v.is_eligible if v else False
    
    # =========================================================================
    # VALIDATOR SELECTION (Stake-Weighted Random)
    # =========================================================================
    
    def select_validators(
        self,
        exclude_node_id: str = "",
        num_validators: int = None,
    ) -> List[ValidatorInfo]:
        """
        Select validators for proof validation using stake-weighted random selection.
        
        Algorithm (from whitepaper Section 7.3):
        =========================================
        Score_i = Stake_i × (1 - r) + Random_i × r × MaxStake
        
        Where r = 0.3 (30% randomness factor)
        
        This ensures:
        - Higher stake = higher chance of selection (security)
        - Randomness prevents stake monopolies (fairness)
        - Small stakers still get validation opportunities (decentralization)
        
        Args:
            exclude_node_id: Node to exclude (the proof submitter)
            num_validators: Number to select (default: self.num_validators_per_proof)
            
        Returns:
            List of selected ValidatorInfo objects
        """
        num_validators = num_validators or self.num_validators_per_proof
        
        with self._validators_lock:
            # Get eligible validators, excluding the submitter
            eligible = [
                v for v in self._validators.values()
                if v.is_eligible and v.node_id != exclude_node_id
            ]
        
        if not eligible:
            logger.warning("No eligible validators available for selection")
            return []
        
        if len(eligible) <= num_validators:
            # Not enough validators, return all
            return eligible
        
        # Calculate selection scores (stake-weighted with randomness)
        max_stake = max(v.stake for v in eligible)
        
        scored = []
        for v in eligible:
            # Score = stake * (1 - randomness) + random * randomness * max_stake
            stake_component = v.stake * (1 - VALIDATOR_SELECTION_RANDOMNESS)
            random_component = random.random() * VALIDATOR_SELECTION_RANDOMNESS * max_stake
            score = stake_component + random_component
            scored.append((v, score))
        
        # Sort by score and select top N
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [v for v, _ in scored[:num_validators]]
        
        logger.debug(f"Selected {len(selected)} validators: "
                    f"{[v.node_id[:12] + '...' for v in selected]}")
        
        return selected
    
    # =========================================================================
    # PROOF SUBMISSION AND CONSENSUS
    # =========================================================================
    
    def submit_proof_for_consensus(
        self,
        proof_signature: str,
        submitter_node_id: str,
    ) -> ConsensusResult:
        """
        Submit a proof for validator consensus.
        
        This initiates the consensus process:
        1. Select validators (stake-weighted)
        2. Request votes from selected validators
        3. Wait for consensus or timeout
        
        Args:
            proof_signature: Unique signature of the proof
            submitter_node_id: Node that submitted the proof
            
        Returns:
            ConsensusResult with current state
        """
        # Check if already pending
        with self._pending_lock:
            if proof_signature in self._pending:
                return self._pending[proof_signature]
        
        # Select validators
        selected = self.select_validators(exclude_node_id=submitter_node_id)
        
        if not selected:
            # No validators available - bootstrap mode
            # In bootstrap (< 2 validators), proofs are auto-accepted
            logger.info(f"Bootstrap mode: Auto-accepting proof {proof_signature[:16]}... "
                       f"(no validators available)")
            result = ConsensusResult(
                proof_signature=proof_signature,
                state=ConsensusState.ACCEPTED,
                consensus_result=True,
                reward_credited=True,
                finalized_at=time.time(),
            )
            return result
        
        # Create pending consensus
        result = ConsensusResult(
            proof_signature=proof_signature,
            state=ConsensusState.COLLECTING_VOTES,
        )
        
        with self._pending_lock:
            self._pending[proof_signature] = result
        
        with self._votes_lock:
            self._votes[proof_signature] = []
        
        # Request votes from selected validators
        # This is done via gossip/gRPC in the actual implementation
        logger.info(f"Proof {proof_signature[:16]}... submitted for consensus "
                   f"({len(selected)} validators selected)")
        
        return result
    
    def cast_vote(
        self,
        proof_signature: str,
        vote: bool,
        reason: str = "",
        validator_id: str = None,
        stake: float = None,
        signature: str = "",
    ) -> Tuple[bool, str]:
        """
        Cast a vote on a proof.
        
        Called by validators when they verify a proof.
        
        Args:
            proof_signature: The proof being voted on
            vote: True = valid, False = invalid
            reason: Explanation for the vote
            validator_id: Validator's node ID (default: self.node_id)
            stake: Validator's stake (default: looked up from registry)
            signature: ECDSA signature of the vote
            
        Returns:
            (success, message)
        """
        validator_id = validator_id or self.node_id
        
        # Get validator info
        with self._validators_lock:
            validator = self._validators.get(validator_id)
            if not validator or not validator.is_eligible:
                return False, f"Node {validator_id[:16]}... is not an eligible validator"
            stake = stake or validator.stake
        
        # Check if proof is pending
        with self._pending_lock:
            result = self._pending.get(proof_signature)
            if not result:
                return False, f"Proof {proof_signature[:16]}... not found or already finalized"
            if result.state not in [ConsensusState.PENDING, ConsensusState.COLLECTING_VOTES]:
                return False, f"Proof already in state: {result.state.value}"
        
        # Create vote
        vote_obj = ValidationVote(
            validator_id=validator_id,
            stake=stake,
            vote=vote,
            reason=reason,
            timestamp=time.time(),
            signature=signature,
        )
        
        # Record vote
        with self._votes_lock:
            if proof_signature not in self._votes:
                self._votes[proof_signature] = []
            
            # Check for duplicate vote
            existing_votes = self._votes[proof_signature]
            if any(v.validator_id == validator_id for v in existing_votes):
                return False, f"Validator {validator_id[:16]}... already voted"
            
            self._votes[proof_signature].append(vote_obj)
        
        # Update validator stats
        with self._validators_lock:
            if validator_id in self._validators:
                self._validators[validator_id].proofs_validated += 1
        
        logger.info(f"Vote recorded: {validator_id[:16]}... voted "
                   f"{'VALID' if vote else 'INVALID'} on {proof_signature[:16]}... "
                   f"(stake={stake:.2f})")
        
        # Check if consensus reached
        self._check_consensus(proof_signature)
        
        return True, f"Vote recorded (stake={stake:.2f})"
    
    def _check_consensus(self, proof_signature: str) -> Optional[ConsensusResult]:
        """
        Check if consensus has been reached for a proof.
        
        Consensus is reached when:
        - At least 2 validators have voted (minimum quorum)
        - AND valid_stake / total_stake >= 66%, OR
        - AND invalid_stake / total_stake >= 66%
        
        Returns updated ConsensusResult if consensus reached.
        """
        with self._pending_lock:
            result = self._pending.get(proof_signature)
            if not result or result.state in [ConsensusState.ACCEPTED, ConsensusState.REJECTED]:
                return result
        
        with self._votes_lock:
            votes = self._votes.get(proof_signature, [])
        
        # Calculate stake-weighted totals
        valid_stake = sum(v.stake for v in votes if v.vote)
        invalid_stake = sum(v.stake for v in votes if not v.vote)
        total_stake = valid_stake + invalid_stake
        
        # Update result
        with self._pending_lock:
            result = self._pending[proof_signature]
            result.total_votes = len(votes)
            result.valid_votes = sum(1 for v in votes if v.vote)
            result.invalid_votes = sum(1 for v in votes if not v.vote)
            result.valid_stake = valid_stake
            result.invalid_stake = invalid_stake
            result.total_stake = total_stake
            
            if total_stake == 0:
                return result
            
            valid_ratio = valid_stake / total_stake
            
            # MINIMUM QUORUM: Need at least 2 validators to vote before consensus
            # This prevents a single validator from unilaterally deciding
            min_votes = min(2, self.num_validators_per_proof)
            if len(votes) < min_votes:
                # Not enough votes yet - keep collecting
                return result
            
            # Check consensus threshold (66%)
            if valid_ratio >= VALIDATION_CONSENSUS_THRESHOLD:
                result.state = ConsensusState.ACCEPTED
                result.consensus_result = True
                result.finalized_at = time.time()
                logger.info(f"Consensus REACHED for {proof_signature[:16]}...: "
                           f"ACCEPTED ({valid_ratio:.1%} valid stake)")
                
            elif (1 - valid_ratio) >= VALIDATION_CONSENSUS_THRESHOLD:
                result.state = ConsensusState.REJECTED
                result.consensus_result = False
                result.finalized_at = time.time()
                logger.info(f"Consensus REACHED for {proof_signature[:16]}...: "
                           f"REJECTED ({1 - valid_ratio:.1%} invalid stake)")
            
            # If consensus reached, process results
            if result.consensus_result is not None:
                self._process_consensus_result(result, votes)
            
            return result
    
    def _process_consensus_result(
        self,
        result: ConsensusResult,
        votes: List[ValidationVote],
    ):
        """
        Process the outcome of consensus.
        
        1. Credit validation fees to validators
        2. Slash validators who voted against consensus (2x penalty)
        3. Update validator accuracy stats
        """
        consensus_vote = result.consensus_result
        
        for vote in votes:
            with self._validators_lock:
                validator = self._validators.get(vote.validator_id)
                if not validator:
                    continue
                
                if vote.vote == consensus_vote:
                    # Correct vote - credit fee and update stats
                    validator.correct_votes += 1
                    
                    # Credit validation fee via ledger
                    if self.ledger:
                        try:
                            # Note: In production, this would be a proper transaction
                            logger.debug(f"Crediting {VALIDATION_FEE_PER_PROOF} NEURO to "
                                        f"validator {vote.validator_id[:16]}...")
                        except Exception as e:
                            logger.warning(f"Failed to credit validation fee: {e}")
                    
                else:
                    # Wrong vote - slash validator
                    validator.wrong_votes += 1
                    result.validators_slashed.append(vote.validator_id)
                    
                    if self.ledger:
                        try:
                            slash_amount = SLASH_AMOUNT * VALIDATOR_SLASH_MULTIPLIER
                            logger.warning(f"Slashing validator {vote.validator_id[:16]}... "
                                          f"{slash_amount} NEURO for voting against consensus")
                            # Note: Actual slashing done via ledger.slash_bad_validator()
                        except Exception as e:
                            logger.warning(f"Failed to slash validator: {e}")
        
        # Trigger callback if set
        if self._on_consensus_reached:
            try:
                self._on_consensus_reached(result)
            except Exception as e:
                logger.error(f"Consensus callback error: {e}")
    
    # =========================================================================
    # CONSENSUS STATUS QUERIES
    # =========================================================================
    
    def get_consensus_status(self, proof_signature: str) -> Optional[ConsensusResult]:
        """Get the current consensus status for a proof."""
        with self._pending_lock:
            return self._pending.get(proof_signature)
    
    def get_pending_count(self) -> int:
        """Get count of proofs pending consensus."""
        with self._pending_lock:
            return sum(
                1 for r in self._pending.values()
                if r.state in [ConsensusState.PENDING, ConsensusState.COLLECTING_VOTES]
            )
    
    def cleanup_old_pending(self, max_age_seconds: float = 300.0):
        """Remove old pending consensus that timed out."""
        cutoff = time.time() - max_age_seconds
        
        with self._pending_lock:
            to_remove = [
                sig for sig, result in self._pending.items()
                if result.created_at < cutoff and
                result.state in [ConsensusState.PENDING, ConsensusState.COLLECTING_VOTES]
            ]
            
            for sig in to_remove:
                result = self._pending[sig]
                result.state = ConsensusState.TIMEOUT
                result.finalized_at = time.time()
                logger.warning(f"Consensus timeout for {sig[:16]}... "
                              f"(votes: {result.total_votes})")
    
    # =========================================================================
    # DYNAMIC STAKE REQUIREMENTS
    # =========================================================================
    
    def get_current_stake_requirement(self) -> float:
        """
        Get the current stake requirement for validators.
        
        This is dynamic based on network size:
        - 0-2 validators: 0 NEURO (bootstrap)
        - 3-10 validators: 10 NEURO
        - 11-50 validators: 50 NEURO
        - 50+ validators: 100 NEURO
        """
        num_validators = self.get_validator_count()
        return get_dynamic_validator_stake(num_validators)
    
    def should_use_consensus(self) -> bool:
        """
        Check if we should use consensus for proof validation.
        
        Returns False during bootstrap (< 2 eligible validators).
        During bootstrap, proofs are auto-accepted.
        """
        return self.get_validator_count() >= 2
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def set_consensus_callback(self, callback: Callable[[ConsensusResult], None]):
        """
        Set callback for when consensus is reached.
        
        The callback receives the ConsensusResult.
        """
        self._on_consensus_reached = callback
    
    def set_verify_callback(self, callback: Callable[[Any], Tuple[bool, str]]):
        """
        Set callback for local proof verification.
        
        The callback receives a proof and returns (is_valid, reason).
        """
        self._verify_proof_locally = callback
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get consensus system statistics."""
        with self._validators_lock:
            validators = list(self._validators.values())
        
        eligible = [v for v in validators if v.is_eligible]
        
        with self._pending_lock:
            pending_count = sum(
                1 for r in self._pending.values()
                if r.state in [ConsensusState.PENDING, ConsensusState.COLLECTING_VOTES]
            )
            accepted_count = sum(
                1 for r in self._pending.values()
                if r.state == ConsensusState.ACCEPTED
            )
            rejected_count = sum(
                1 for r in self._pending.values()
                if r.state == ConsensusState.REJECTED
            )
        
        return {
            "total_registered": len(validators),
            "eligible_validators": len(eligible),
            "total_stake": sum(v.stake for v in eligible),
            "avg_stake": sum(v.stake for v in eligible) / len(eligible) if eligible else 0,
            "proofs_pending": pending_count,
            "proofs_accepted": accepted_count,
            "proofs_rejected": rejected_count,
            "stake_requirement": self.get_current_stake_requirement(),
            "using_consensus": self.should_use_consensus(),
            "consensus_threshold": VALIDATION_CONSENSUS_THRESHOLD,
            "validators_per_proof": self.num_validators_per_proof,
        }
