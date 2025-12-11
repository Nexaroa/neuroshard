"""
NeuroShard Consensus Module

This module implements the Hybrid Validator Consensus system from the whitepaper:

1. **ProofVerifier**: Universal constraints (hardware limits, format validation)
2. **ValidatorConsensus**: Stake-weighted proof validation with 66% threshold

Architecture:
=============
- Validators = Nodes with Last Layer + LM Head + 100 NEURO staked
- Multiple validators verify each proof via stake-weighted voting
- 66% stake threshold required for consensus
- Validators earn 0.001 NEURO per proof validated
- Bad validators (vote against consensus) are slashed at 2x rate

See: docs/whitepaper/neuroshard_whitepaper.tex Section 7 (Hybrid Validator System)
"""

from neuroshard.core.consensus.verifier import ProofVerifier
from neuroshard.core.consensus.validator_consensus import (
    ValidatorConsensus,
    ValidatorInfo,
    ValidationVote,
    ConsensusResult,
)

__all__ = [
    "ProofVerifier",
    "ValidatorConsensus", 
    "ValidatorInfo",
    "ValidationVote",
    "ConsensusResult",
]
