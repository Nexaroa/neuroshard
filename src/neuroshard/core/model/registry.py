"""
Dynamic Tokenizer Registry

Allows tokenizer vocabulary to grow as network needs evolve.

FUTURE ENHANCEMENT:
- Start with 32k vocabulary (English-focused)
- Grow to 64k when multilingual support needed
- Grow to 128k when code/math support needed

Just like architecture, tokenizer can be upgraded!
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for NeuroShard tokenizer."""
    vocab_size: int
    tokenizer_version: int
    model_path: str
    supported_languages: list
    
    def estimate_embedding_memory_mb(self, hidden_dim: int) -> float:
        """Calculate memory for embedding layer with this vocab size."""
        # Embedding: vocab_size × hidden_dim
        # Plus gradients and optimizer states (×4 total)
        params = self.vocab_size * hidden_dim
        return (params * 4 * 4) / (1024 * 1024)
    
    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "tokenizer_version": self.tokenizer_version,
            "model_path": self.model_path,
            "supported_languages": self.supported_languages,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# Default tokenizer configurations
TOKENIZER_CONFIGS = {
    # Version 1: English-focused (bootstrap)
    1: TokenizerConfig(
        vocab_size=32000,
        tokenizer_version=1,
        model_path="neuroshard_tokenizer_v1.model",
        supported_languages=["en"],
    ),
    
    # Version 2: Multilingual (future upgrade)
    # Triggered when network has >1000 nodes from diverse regions
    2: TokenizerConfig(
        vocab_size=64000,
        tokenizer_version=2,
        model_path="neuroshard_tokenizer_v2.model",
        supported_languages=["en", "es", "fr", "de", "zh", "ja"],
    ),
    
    # Version 3: Code + Math specialized (future upgrade)
    # Triggered by community vote or when code training data > 30%
    3: TokenizerConfig(
        vocab_size=100000,
        tokenizer_version=3,
        model_path="neuroshard_tokenizer_v3.model",
        supported_languages=["en", "code", "math"],
    ),
}


def get_current_tokenizer_config(network_size: int = 1) -> TokenizerConfig:
    """
    Get appropriate tokenizer config for current network state.
    
    Upgrade triggers:
    - v1 → v2: Network has >1000 nodes (multilingual needed)
    - v2 → v3: Community votes for code specialization
    
    Args:
        network_size: Number of active nodes
    
    Returns:
        TokenizerConfig for current network state
    """
    if network_size < 1000:
        # Bootstrap: English-focused 32k vocab
        return TOKENIZER_CONFIGS[1]
    elif network_size < 5000:
        # Growth: Multilingual 64k vocab
        # TODO: Implement migration from v1 → v2
        logger.warning("Network ready for multilingual tokenizer upgrade (v1 → v2)")
        return TOKENIZER_CONFIGS[1]  # Stay on v1 until migration implemented
    else:
        # Maturity: Specialized 100k vocab
        logger.warning("Network ready for code-specialized tokenizer (v2 → v3)")
        return TOKENIZER_CONFIGS[2]  # Stay on v2 until migration implemented


def should_upgrade_tokenizer(
    current: TokenizerConfig,
    new: TokenizerConfig,
) -> tuple[bool, str]:
    """
    Determine if tokenizer upgrade is worthwhile.
    
    Tokenizer upgrades are EXPENSIVE (require retraining):
    - All embeddings must be expanded (vocab_size × hidden_dim)
    - All existing checkpoints incompatible
    - Requires community vote (unlike architecture, which is automatic)
    
    Returns:
        (should_upgrade, reason)
    """
    if new.tokenizer_version <= current.tokenizer_version:
        return False, "Not a newer version"
    
    # Tokenizer upgrades require community governance vote
    # Unlike architecture (automatic), tokenizer affects all model outputs
    reason = (f"Tokenizer upgrade available: v{current.tokenizer_version} → v{new.tokenizer_version} "
              f"({current.vocab_size} → {new.vocab_size} tokens). "
              f"Requires NeuroDAO vote and coordinated upgrade.")
    
    return True, reason


# INTEGRATION WITH ARCHITECTURE

def adjust_architecture_for_tokenizer(
    arch: 'ModelArchitecture',  # type: ignore
    tokenizer: TokenizerConfig
) -> 'ModelArchitecture':  # type: ignore
    """
    Adjust architecture to account for tokenizer vocab size.
    
    Larger vocab → larger embedding → less memory for layers.
    """
    from neuroshard.core.model.scaler import ModelArchitecture
    
    # Estimate embedding memory
    embedding_mem_mb = tokenizer.estimate_embedding_memory_mb(arch.hidden_dim)
    
    # If embedding is too large, we might need to reduce num_layers slightly
    # (This is handled in calculate_optimal_architecture already, but good to verify)
    
    return ModelArchitecture(
        hidden_dim=arch.hidden_dim,
        intermediate_dim=arch.intermediate_dim,
        num_layers=arch.num_layers,
        num_heads=arch.num_heads,
        num_kv_heads=arch.num_kv_heads,
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab_size!
        max_seq_len=arch.max_seq_len,
        dropout=arch.dropout,
        rope_theta=arch.rope_theta,
    )

