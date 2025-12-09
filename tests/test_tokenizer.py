#!/usr/bin/env python3
"""
Test BPE Tokenizer for NeuroShard

Tests the NeuroTokenizer implementation to ensure:
1. Basic encoding/decoding works
2. BPE merges are learned correctly
3. Tokenizer serialization works
4. Edge cases are handled
"""

import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from neuroshard.core.model.tokenizer import NeuroTokenizer, get_neuro_tokenizer, reset_tokenizer


class TestNeuroTokenizerBasic(unittest.TestCase):
    """Test basic tokenizer functionality."""
    
    def setUp(self):
        """Create a fresh tokenizer for each test."""
        reset_tokenizer()
        self.tokenizer = NeuroTokenizer()
    
    def test_initial_vocab_size(self):
        """Test that initial vocab size is 266 (10 special + 256 bytes)."""
        self.assertEqual(self.tokenizer.current_vocab_size, 266)
        self.assertEqual(len(self.tokenizer), 266)
    
    def test_special_tokens(self):
        """Test special token IDs."""
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.bos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(self.tokenizer.unk_token_id, 3)
    
    def test_encode_simple(self):
        """Test basic encoding without BPE merges."""
        text = "hello"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Should be byte-level tokens: h=104, e=101, l=108, l=108, o=111
        # Offset by 10: 114, 111, 118, 118, 121
        expected = [114, 111, 118, 118, 121]
        self.assertEqual(tokens, expected)
    
    def test_encode_with_special_tokens(self):
        """Test encoding with BOS/EOS tokens."""
        text = "hi"
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # BOS + h(104+10) + i(105+10) + EOS = 1, 114, 115, 2
        self.assertEqual(tokens[0], 1)  # BOS
        self.assertEqual(tokens[-1], 2)  # EOS
        self.assertEqual(len(tokens), 4)
    
    def test_decode_simple(self):
        """Test basic decoding."""
        # Encode then decode should roundtrip
        text = "Hello, World!"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text)
    
    def test_decode_with_special_tokens(self):
        """Test decoding skips special tokens."""
        text = "test"
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(decoded, text)
    
    def test_unicode_roundtrip(self):
        """Test encoding/decoding Unicode text."""
        texts = [
            "Hello 世界",
            "Привет мир",
            "🚀 NeuroShard",
            "日本語テスト",
        ]
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            decoded = self.tokenizer.decode(tokens)
            self.assertEqual(decoded, text, f"Failed for: {text}")


class TestBPELearning(unittest.TestCase):
    """Test BPE merge learning."""
    
    def setUp(self):
        reset_tokenizer()
        self.tokenizer = NeuroTokenizer()
    
    def test_learn_merges_increases_vocab(self):
        """Test that learning merges increases vocabulary."""
        initial_vocab = self.tokenizer.current_vocab_size
        
        # Create repetitive text to learn merges from
        texts = ["the cat sat on the mat. " * 100] * 10
        
        self.tokenizer.learn_merges(texts, num_merges=50, min_frequency=2)
        
        # Vocabulary should have grown
        self.assertGreater(self.tokenizer.current_vocab_size, initial_vocab)
        self.assertGreater(len(self.tokenizer.merges), 0)
    
    def test_merges_compress_text(self):
        """Test that learned merges compress repeated patterns."""
        # Before learning
        text = "the the the the"
        tokens_before = self.tokenizer.encode(text, add_special_tokens=False)
        len_before = len(tokens_before)
        
        # Learn merges from repetitive text
        texts = ["the quick brown fox " * 100] * 20
        self.tokenizer.learn_merges(texts, num_merges=100, min_frequency=2)
        
        # After learning - should be shorter
        tokens_after = self.tokenizer.encode(text, add_special_tokens=False)
        len_after = len(tokens_after)
        
        self.assertLess(len_after, len_before, 
            f"Expected compression: {len_before} -> {len_after}")
    
    def test_decode_after_merges(self):
        """Test that decoding still works after learning merges."""
        texts = ["hello world " * 100] * 10
        self.tokenizer.learn_merges(texts, num_merges=20, min_frequency=2)
        
        # Roundtrip should still work
        test_text = "hello world hello world"
        tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, test_text)
    
    def test_max_vocab_respected(self):
        """Test that vocab size limit is respected."""
        small_tokenizer = NeuroTokenizer(vocab_size=300)  # Only 34 merges allowed
        texts = ["the quick brown fox " * 100] * 20
        
        small_tokenizer.learn_merges(texts, num_merges=100, min_frequency=2)
        
        # Should not exceed vocab_size
        self.assertLessEqual(small_tokenizer.current_vocab_size, 300)


class TestTokenizerSerialization(unittest.TestCase):
    """Test tokenizer save/load functionality."""
    
    def setUp(self):
        reset_tokenizer()
        self.tokenizer = NeuroTokenizer()
    
    def test_save_load_roundtrip(self):
        """Test that tokenizer can be saved and loaded."""
        # Learn some merges
        texts = ["hello world " * 100] * 10
        self.tokenizer.learn_merges(texts, num_merges=20, min_frequency=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(path)
            
            # Load into new tokenizer
            loaded = NeuroTokenizer.load(path)
            
            # Should have same merges
            self.assertEqual(len(loaded.merges), len(self.tokenizer.merges))
            self.assertEqual(loaded.current_vocab_size, self.tokenizer.current_vocab_size)
            
            # Should encode/decode the same
            test_text = "hello world test"
            orig_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            loaded_tokens = loaded.encode(test_text, add_special_tokens=False)
            self.assertEqual(orig_tokens, loaded_tokens)
    
    def test_load_nonexistent_creates_new(self):
        """Test that loading nonexistent file creates new tokenizer."""
        loaded = NeuroTokenizer.load("/nonexistent/path/tokenizer.json")
        self.assertEqual(loaded.current_vocab_size, 266)
    
    def test_json_format(self):
        """Test that saved tokenizer is valid JSON."""
        texts = ["test " * 100] * 5
        self.tokenizer.learn_merges(texts, num_merges=10, min_frequency=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(path)
            
            with open(path) as f:
                data = json.load(f)
            
            # Check required fields
            self.assertIn("vocab_size", data)
            self.assertIn("next_merge_id", data)
            self.assertIn("merges", data)


class TestTokenizerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        reset_tokenizer()
        self.tokenizer = NeuroTokenizer()
    
    def test_empty_string(self):
        """Test encoding empty string."""
        tokens = self.tokenizer.encode("", add_special_tokens=False)
        self.assertEqual(tokens, [])
        
        tokens_with_special = self.tokenizer.encode("", add_special_tokens=True)
        self.assertEqual(tokens_with_special, [1, 2])  # Just BOS, EOS
    
    def test_truncation(self):
        """Test truncation."""
        text = "hello world"
        tokens = self.tokenizer.encode(text, add_special_tokens=True, 
                                       max_length=5, truncation=True)
        self.assertEqual(len(tokens), 5)
    
    def test_padding(self):
        """Test padding."""
        text = "hi"
        tokens = self.tokenizer.encode(text, add_special_tokens=True,
                                       max_length=10, padding=True)
        self.assertEqual(len(tokens), 10)
        self.assertEqual(tokens[-1], 0)  # PAD token
    
    def test_batch_encode(self):
        """Test batch encoding."""
        texts = ["hello", "world", "test"]
        result = self.tokenizer.batch_encode(texts, max_length=10, 
                                             padding=True, truncation=True)
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(len(result["input_ids"]), 3)
        
        # All should be padded to same length
        for ids in result["input_ids"]:
            self.assertEqual(len(ids), 10)
    
    def test_source_contribution_tracking(self):
        """Test that source contributions are tracked."""
        texts = ["test " * 100] * 5
        
        self.assertFalse(self.tokenizer.has_source_contributed("test_source"))
        
        self.tokenizer.learn_merges(texts, num_merges=10, min_frequency=2)
        self.tokenizer.record_source_contribution("test_source", 10)
        
        self.assertTrue(self.tokenizer.has_source_contributed("test_source"))
        
        stats = self.tokenizer.get_stats()
        self.assertIn("sources_contributed", stats)
        self.assertEqual(stats["sources_contributed"]["test_source"], 10)


class TestLearnedTokenizer(unittest.TestCase):
    """Test the actual learned tokenizer from scripts/."""
    
    def test_load_learned_tokenizer(self):
        """Test loading the pre-learned tokenizer."""
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "learned_tokenizer.json"
        )
        
        if not os.path.exists(tokenizer_path):
            self.skipTest("Learned tokenizer not found")
        
        tokenizer = NeuroTokenizer.load(tokenizer_path)
        
        # Should have BPE merges
        self.assertGreater(len(tokenizer.merges), 0)
        self.assertGreater(tokenizer.current_vocab_size, 266)
        
        # Should have fineweb-edu contribution
        self.assertTrue(tokenizer.has_source_contributed("fineweb-edu"))
    
    def test_learned_tokenizer_roundtrip(self):
        """Test roundtrip with learned tokenizer."""
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "learned_tokenizer.json"
        )
        
        if not os.path.exists(tokenizer_path):
            self.skipTest("Learned tokenizer not found")
        
        tokenizer = NeuroTokenizer.load(tokenizer_path)
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "NeuroShard enables decentralized AI training.",
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            self.assertEqual(decoded, text, f"Roundtrip failed for: {text}")


class TestDataFlowConsistency(unittest.TestCase):
    """Test that tokenization is consistent across the data flow."""
    
    def test_training_shard_format(self):
        """
        Verify that shards would be created in the correct format.
        
        Training shards should be token IDs without special tokens.
        """
        tokenizer = NeuroTokenizer()
        
        # Simulate what populate_genesis_s3.py does
        text = "This is a sample document for training."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Should be plain token IDs
        self.assertIsInstance(tokens, list)
        for t in tokens:
            self.assertIsInstance(t, int)
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, tokenizer.current_vocab_size)
        
        # Should NOT have BOS/EOS
        self.assertNotEqual(tokens[0], 1)  # Not BOS
        self.assertNotEqual(tokens[-1], 2)  # Not EOS
    
    def test_inference_format(self):
        """
        Verify inference tokenization includes special tokens.
        """
        tokenizer = NeuroTokenizer()
        
        text = "Hello, how are you?"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Should have BOS at start, EOS at end
        self.assertEqual(tokens[0], 1)  # BOS
        self.assertEqual(tokens[-1], 2)  # EOS


if __name__ == "__main__":
    unittest.main(verbosity=2)
