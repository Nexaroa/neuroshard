"""
NeuroLLM Tokenizer

A BPE (Byte Pair Encoding) tokenizer for NeuroLLM that is trained from scratch
by the network itself.
tokenizer - it's a truly decentralized tokenizer that grows with the network.

The tokenizer starts with a base vocabulary (bytes + special tokens) and learns
new subword units as more training data is contributed by the network.

Features:
- Pure BPE implementation (no external dependencies for core functionality)
- Starts with byte-level vocabulary (256 tokens)
- Learns merges from contributed training data
- Can be updated through network consensus
- Fully serializable for checkpoint distribution
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuroTokenizer:
    """
    A truly decentralized BPE tokenizer for NeuroLLM.
    
    Unlike traditional tokenizers that are pre-trained on massive corpora,
    this tokenizer starts with a minimal vocabulary and learns from the
    training data contributed by network participants.
    """
    
    # Special tokens (reserved IDs 0-9)
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"
    
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    
    # Byte tokens start at ID 10 (256 bytes = IDs 10-265)
    BYTE_OFFSET = 10
    
    # Learned merges start at ID 266
    MERGE_OFFSET = 266
    
    def __init__(self, vocab_size: int = 32000):
        """
        Initialize the NeuroLLM tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (default 32000)
        """
        self.vocab_size = vocab_size
        
        # Core vocabulary
        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        
        # Byte vocabulary (256 bytes)
        self.byte_to_id = {i: i + self.BYTE_OFFSET for i in range(256)}
        self.id_to_byte = {v: k for k, v in self.byte_to_id.items()}
        
        # Learned BPE merges: (token1, token2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_to_tokens: Dict[int, Tuple[int, int]] = {}  # Reverse lookup
        
        # Token to string (for decoding merged tokens)
        self.id_to_string: Dict[int, str] = {}
        
        # Next available ID for new merges
        self.next_merge_id = self.MERGE_OFFSET
        
        # Track which data sources have contributed merges
        # Format: {"source_name": num_merges_contributed}
        self.sources_contributed: Dict[str, int] = {}
        
        # Statistics
        self.total_tokens_processed = 0
        
        logger.info(f"NeuroTokenizer initialized with vocab_size={vocab_size}")
    
    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID
    
    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID
    
    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID
    
    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID
    
    @property
    def current_vocab_size(self) -> int:
        """
        The current vocabulary size (valid token IDs: 0 to current_vocab_size-1).
        
        This grows as the tokenizer learns BPE merges:
        - Initial: 266 (10 special + 256 bytes)
        - After learning: 266 + num_merges
        - Maximum: vocab_size (32000)
        
        IMPORTANT: During inference, only tokens 0 to current_vocab_size-1 are valid.
        Tokens beyond this have no learned representation and should not be sampled.
        """
        return self.next_merge_id
    
    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to byte-level token IDs."""
        return [self.byte_to_id[b] for b in text.encode('utf-8')]
    
    def _apply_merges(self, token_ids: List[int]) -> List[int]:
        """Apply learned BPE merges to a sequence of token IDs."""
        if not self.merges:
            return token_ids
        
        # Iteratively apply merges (greedy)
        while len(token_ids) > 1:
            # Find the best merge (lowest ID = learned earliest = most frequent)
            best_merge = None
            best_idx = -1
            best_merged_id = float('inf')
            
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merges:
                    merged_id = self.merges[pair]
                    if merged_id < best_merged_id:
                        best_merge = pair
                        best_idx = i
                        best_merged_id = merged_id
            
            if best_merge is None:
                break
            
            # Apply the merge
            token_ids = (
                token_ids[:best_idx] + 
                [best_merged_id] + 
                token_ids[best_idx + 2:]
            )
        
        return token_ids
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length (truncate if longer)
            truncation: Whether to truncate
            padding: Whether to pad to max_length
            
        Returns:
            List of token IDs
        """
        # Convert to bytes
        byte_ids = self._text_to_bytes(text)
        
        # Apply BPE merges
        token_ids = self._apply_merges(byte_ids)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.BOS_ID] + token_ids + [self.EOS_ID]
        
        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Padding
        if padding and max_length and len(token_ids) < max_length:
            token_ids = token_ids + [self.PAD_ID] * (max_length - len(token_ids))
        
        self.total_tokens_processed += len(token_ids)
        return token_ids
    
    def _decode_token(self, token_id: int) -> bytes:
        """Decode a single token ID to bytes."""
        # Special tokens
        if token_id in [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]:
            return b''
        
        # Byte token
        if token_id in self.id_to_byte:
            return bytes([self.id_to_byte[token_id]])
        
        # Merged token - recursively decode
        if token_id in self.merge_to_tokens:
            t1, t2 = self.merge_to_tokens[token_id]
            return self._decode_token(t1) + self._decode_token(t2)
        
        # Unknown token - this should NOT happen in normal operation
        # If we get here, the model output a token ID beyond current_vocab_size
        # This is a bug in the generation code (should be masking invalid tokens)
        logger.warning(f"Unknown token ID {token_id} (vocab_size={self.current_vocab_size}) - using UNK")
        return b'<unk>'
    
    def decode(
        self, 
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        byte_sequence = b''
        
        for tid in token_ids:
            if skip_special_tokens and tid in [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]:
                continue
            byte_sequence += self._decode_token(tid)
        
        # Decode UTF-8, replacing errors
        return byte_sequence.decode('utf-8', errors='replace')
    
    def learn_merges(self, texts: List[str], num_merges: int = 1000, min_frequency: int = 2):
        """
        Learn new BPE merges from training data.
        
        This is called during training to expand the vocabulary based on
        the data contributed by network participants.
        
        Args:
            texts: List of training texts
            num_merges: Number of new merges to learn
            min_frequency: Minimum pair frequency to create merge
        """
        if self.next_merge_id + num_merges > self.vocab_size:
            num_merges = self.vocab_size - self.next_merge_id
            if num_merges <= 0:
                logger.warning("Vocabulary is full, cannot learn more merges")
                return
        
        # Tokenize all texts to current vocabulary
        all_sequences = []
        for text in texts:
            byte_ids = self._text_to_bytes(text)
            token_ids = self._apply_merges(byte_ids)
            all_sequences.append(token_ids)
        
        merges_learned = 0
        
        while merges_learned < num_merges:
            # Count all adjacent pairs
            pair_counts = Counter()
            for seq in all_sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    if pair not in self.merges:  # Don't count already merged pairs
                        pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair, count = pair_counts.most_common(1)[0]
            
            if count < min_frequency:
                break
            
            # Create new merge
            new_id = self.next_merge_id
            self.merges[best_pair] = new_id
            self.merge_to_tokens[new_id] = best_pair
            self.next_merge_id += 1
            merges_learned += 1
            
            # Apply merge to all sequences
            for i, seq in enumerate(all_sequences):
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and (seq[j], seq[j + 1]) == best_pair:
                        new_seq.append(new_id)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                all_sequences[i] = new_seq
        
        logger.info(f"Learned {merges_learned} new merges, vocab size now {len(self)}")
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        input_ids = []
        attention_mask = []
        
        for text in texts:
            ids = self.encode(text, max_length=max_length, truncation=truncation, padding=padding)
            input_ids.append(ids)
            attention_mask.append([1 if tid != self.PAD_ID else 0 for tid in ids])
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def save(self, path: str):
        """
        Save tokenizer to a JSON file.
        
        Args:
            path: Path to save the tokenizer JSON file (must end with .json)
        """
        if not path.endswith('.json'):
            path = path + '.json'
        
        config = {
            "vocab_size": self.vocab_size,
            "next_merge_id": self.next_merge_id,
            "total_tokens_processed": self.total_tokens_processed,
            "sources_contributed": self.sources_contributed,
            # Convert tuple keys to strings for JSON
            "merges": {f"{k[0]}_{k[1]}": v for k, v in self.merges.items()},
        }
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Tokenizer saved to {path} ({self.current_vocab_size} tokens, {len(self.merges)} merges)")
    
    @classmethod
    def load(cls, path: str) -> 'NeuroTokenizer':
        """
        Load tokenizer from a JSON file.
        
        Args:
            path: Path to the tokenizer JSON file
            
        Returns:
            Loaded NeuroTokenizer instance
        """
        if not path.endswith('.json'):
            path = path + '.json'
        
        if not os.path.exists(path):
            logger.warning(f"No tokenizer found at {path}, creating new one")
            return cls()
        
        with open(path) as f:
            config = json.load(f)
        
        tokenizer = cls(vocab_size=config.get("vocab_size", 32000))
        tokenizer.next_merge_id = config.get("next_merge_id", cls.MERGE_OFFSET)
        tokenizer.total_tokens_processed = config.get("total_tokens_processed", 0)
        tokenizer.sources_contributed = config.get("sources_contributed", {})
        
        # Restore merges
        merges_data = config.get("merges", {})
        for key_str, merged_id in merges_data.items():
            t1, t2 = map(int, key_str.split("_"))
            tokenizer.merges[(t1, t2)] = merged_id
            tokenizer.merge_to_tokens[merged_id] = (t1, t2)
        
        logger.info(f"Tokenizer loaded from {path} ({tokenizer.current_vocab_size} tokens, {len(tokenizer.merges)} merges)")
        return tokenizer
    
    def __len__(self) -> int:
        """Return current vocabulary size (valid token count)."""
        # This should match current_vocab_size for consistency
        # Base: 266 tokens (IDs 0-265: 10 special/reserved + 256 bytes)
        # Plus: learned merges
        return self.current_vocab_size
    
    def has_source_contributed(self, source_name: str) -> bool:
        """Check if a data source has already contributed merges."""
        return source_name in self.sources_contributed
    
    def record_source_contribution(self, source_name: str, num_merges: int):
        """Record that a source has contributed merges."""
        self.sources_contributed[source_name] = num_merges
        logger.info(f"Recorded contribution: '{source_name}' added {num_merges} merges")
    
    def get_stats(self) -> Dict:
        """Get tokenizer statistics."""
        return {
            "vocab_size": self.vocab_size,
            "current_vocab": len(self),
            "num_merges": len(self.merges),
            "total_tokens_processed": self.total_tokens_processed,
            "can_learn_more": self.next_merge_id < self.vocab_size,
            "sources_contributed": self.sources_contributed,
        }


# Global tokenizer instance
_tokenizer: Optional[NeuroTokenizer] = None


def get_neuro_tokenizer() -> NeuroTokenizer:
    """Get the global NeuroLLM tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = NeuroTokenizer()
    return _tokenizer


def reset_tokenizer():
    """Reset the global tokenizer (for testing)."""
    global _tokenizer
    _tokenizer = None
