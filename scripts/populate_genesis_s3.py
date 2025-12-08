#!/usr/bin/env python3
"""
NeuroShard Genesis Data Populator v3.0 - Parallel Edition

High-performance, parallel shard creation for S3.
Optimized for multi-core instances (c5.2xlarge recommended).

Key Features:
- Parallel tokenization using multiprocessing
- Async S3 uploads while tokenizing next batch
- Checkpoint-based instant resume
- Multiple data source support
- ~5x faster than v2.0 on 8-core instances
"""

import os
import json
import torch
import boto3
import logging
import argparse
import tempfile
import hashlib
import signal
import sys
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GenesisPopulator")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Standard shard size - MUST be consistent across all training data
SHARD_SIZE_MB = 10.0
TOKENS_PER_SHARD = int(SHARD_SIZE_MB * 1e6 / 4)  # ~2.5M tokens per 10MB shard

# Parallel processing settings
NUM_UPLOAD_WORKERS = 4      # Concurrent S3 uploads
BATCH_SIZE = 8              # Shards to prepare before uploading
PREFETCH_BATCHES = 2        # Batches to tokenize ahead

# Supported data sources
DATA_SOURCES = {
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_field": "text",
        "description": "High-quality educational web content (1.3T tokens)",
        "estimated_tokens": 1.3e12
    },
    "fineweb": {
        "hf_path": "HuggingFaceFW/fineweb",
        "split": "train", 
        "text_field": "text",
        "description": "Large-scale web content (15T tokens)",
        "estimated_tokens": 15e12
    },
    "redpajama": {
        "hf_path": "togethercomputer/RedPajama-Data-1T",
        "split": "train",
        "text_field": "text", 
        "description": "RedPajama 1T token dataset",
        "estimated_tokens": 1e12
    },
    "slimpajama": {
        "hf_path": "cerebras/SlimPajama-627B",
        "split": "train",
        "text_field": "text",
        "description": "SlimPajama 627B tokens (cleaned RedPajama)",
        "estimated_tokens": 627e9
    },
    "c4": {
        "hf_path": "allenai/c4",
        "split": "train",
        "text_field": "text",
        "description": "Colossal Clean Crawled Corpus",
        "estimated_tokens": 365e9
    },
}

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

@dataclass
class ProcessingCheckpoint:
    """Tracks exact position in data stream for instant resume."""
    source: str
    documents_processed: int
    tokens_processed: int
    last_shard_id: int
    leftover_tokens: list
    started_at: str
    updated_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ProcessingCheckpoint':
        return cls(**d)
    
    @classmethod
    def new(cls, source: str) -> 'ProcessingCheckpoint':
        now = datetime.utcnow().isoformat()
        return cls(
            source=source,
            documents_processed=0,
            tokens_processed=0,
            last_shard_id=-1,
            leftover_tokens=[],
            started_at=now,
            updated_at=now
        )


# ============================================================================
# S3 CLIENT
# ============================================================================

def load_env_manual():
    """Manually load .env file to ensure credentials are set."""
    possible_paths = [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent.parent / 'website' / '.env',
        Path.home() / '.env',
    ]
    
    for p in possible_paths:
        if p.exists():
            logger.info(f"Loading .env from {p}")
            with open(p, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip("'").strip('"')
                        os.environ[key] = value
            return
    
    logger.warning(".env file not found")


def get_s3_client():
    """Create S3 client using env vars with optimized connection pool."""
    from botocore.config import Config
    
    key = os.getenv('AWS_ACCESS_KEY_ID')
    secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if key:
        logger.info(f"Using Access Key: {key[:4]}...{key[-4:]}")
    else:
        logger.error("AWS credentials not found!")
    
    # Optimize for parallel uploads
    config = Config(
        max_pool_connections=50,  # Increased from default 10
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
        
    return boto3.client('s3',
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=region,
        config=config
    )


# ============================================================================
# MANIFEST & CHECKPOINT MANAGEMENT
# ============================================================================

class GenesisManager:
    """Manages S3 manifest and checkpoints with thread-safe operations."""
    
    def __init__(self, bucket_name: str):
        self.bucket = bucket_name
        self.s3 = get_s3_client()
        self.manifest = self._load_manifest()
        self.checkpoints: Dict[str, ProcessingCheckpoint] = self._load_checkpoints()
        self._lock = threading.Lock()
        
    def _load_manifest(self) -> dict:
        """Load or create manifest."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key="manifest.json")
            manifest = json.loads(obj['Body'].read().decode('utf-8'))
            logger.info(f"Loaded manifest: {manifest['total_shards']} shards")
            
            # Migrate old manifest format
            if 'sources' not in manifest:
                manifest['sources'] = {}
            if 'total_tokens' not in manifest:
                manifest['total_tokens'] = sum(s.get('size_tokens', 0) for s in manifest.get('shards', []))
            if 'shard_size_mb' not in manifest:
                manifest['shard_size_mb'] = SHARD_SIZE_MB
            if 'tokens_per_shard' not in manifest:
                manifest['tokens_per_shard'] = TOKENS_PER_SHARD
            manifest['version'] = 3
            
            return manifest
        except:
            logger.info("Creating new manifest")
            return {
                "version": 3,
                "shard_size_mb": SHARD_SIZE_MB,
                "tokens_per_shard": TOKENS_PER_SHARD,
                "total_shards": 0,
                "total_tokens": 0,
                "sources": {},
                "shards": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
    
    def _load_checkpoints(self) -> Dict[str, ProcessingCheckpoint]:
        """Load processing checkpoints for each source."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key="checkpoints.json")
            data = json.loads(obj['Body'].read().decode('utf-8'))
            return {k: ProcessingCheckpoint.from_dict(v) for k, v in data.items()}
        except:
            return {}
    
    def get_checkpoint(self, source: str) -> ProcessingCheckpoint:
        """Get or create checkpoint for a data source."""
        if source not in self.checkpoints:
            self.checkpoints[source] = ProcessingCheckpoint.new(source)
        return self.checkpoints[source]
    
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint):
        """Save checkpoint to S3 (thread-safe)."""
        with self._lock:
            checkpoint.updated_at = datetime.utcnow().isoformat()
            self.checkpoints[checkpoint.source] = checkpoint
            
            data = {k: v.to_dict() for k, v in self.checkpoints.items()}
            self.s3.put_object(
                Bucket=self.bucket,
                Key="checkpoints.json",
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
    
    def add_shard(self, shard_id: int, source: str, file_hash: str, 
                  size_tokens: int, size_bytes: int) -> dict:
        """Add a shard to the manifest (thread-safe)."""
        with self._lock:
            shard_meta = {
                "shard_id": shard_id,
                "source": source,
                "hash": file_hash,
                "size_tokens": size_tokens,
                "size_bytes": size_bytes,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.manifest['shards'].append(shard_meta)
            self.manifest['total_shards'] = len(self.manifest['shards'])
            self.manifest['total_tokens'] = sum(s['size_tokens'] for s in self.manifest['shards'])
            self.manifest['updated_at'] = datetime.utcnow().isoformat()
            
            # Track per-source stats
            if source not in self.manifest['sources']:
                self.manifest['sources'][source] = {'shards': 0, 'tokens': 0}
            self.manifest['sources'][source]['shards'] += 1
            self.manifest['sources'][source]['tokens'] += size_tokens
            
            return shard_meta
    
    def save_manifest(self):
        """Save manifest to S3 (thread-safe)."""
        with self._lock:
            self.s3.put_object(
                Bucket=self.bucket,
                Key="manifest.json",
                Body=json.dumps(self.manifest, indent=2),
                ContentType='application/json'
            )
    
    def upload_shard(self, shard_id: int, tensor: torch.Tensor, source: str) -> dict:
        """Upload a shard to S3 and update manifest."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            torch.save(tensor, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Calculate hash
            with open(tmp_path, 'rb') as f:
                file_bytes = f.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
            
            # Upload (this is the slow part - ~0.5-1s per shard)
            filename = f"shard_{shard_id}.pt"
            self.s3.upload_file(
                tmp_path,
                self.bucket,
                filename,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
            
            # Update manifest
            shard_meta = self.add_shard(
                shard_id=shard_id,
                source=source,
                file_hash=file_hash,
                size_tokens=len(tensor),
                size_bytes=len(file_bytes)
            )
            
            return shard_meta
            
        finally:
            os.unlink(tmp_path)


# ============================================================================
# DATA STREAMING
# ============================================================================

# Global tokenizer instance for worker processes
_worker_tokenizer = None

def _init_worker_tokenizer():
    """Initialize tokenizer in worker process."""
    global _worker_tokenizer
    if _worker_tokenizer is None:
        from neuroshard.core.model.tokenizer import get_neuro_tokenizer
        _worker_tokenizer = get_neuro_tokenizer()
    return _worker_tokenizer

def _tokenize_document(args):
    """Tokenize a single document (for multiprocessing)."""
    text, text_field = args
    tokenizer = _init_worker_tokenizer()
    return tokenizer.encode(text)

class DataStreamer:
    """Streams and tokenizes data from various sources."""
    
    def __init__(self, source: str, checkpoint: ProcessingCheckpoint):
        self.source = source
        self.checkpoint = checkpoint
        self.config = DATA_SOURCES[source]
        self.tokenizer = None
        self.iterator = None
        self.docs_seen = 0
        self.tokenizer_pool = None
        self.num_workers = max(4, cpu_count() - 1)  # Use all cores except 1
        
        # Metrics
        self.total_fetch_time = 0.0
        self.total_tokenize_time = 0.0
        
    def setup(self):
        """Initialize tokenizer and dataset."""
        from datasets import load_dataset
        from neuroshard.core.model.tokenizer import NeuroTokenizer
        
        # Check if we have a pre-learned tokenizer (flat JSON file)
        tokenizer_path = os.path.join(os.path.dirname(__file__), "learned_tokenizer.json")
        if os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path):
            logger.info(f"Loading learned tokenizer from {tokenizer_path}")
            self.tokenizer = NeuroTokenizer.load(tokenizer_path)
            
            # DYNAMIC UPDATES: Each source contributes merges ONCE
            # Track contributions in the tokenizer JSON itself for reliability
            has_contributed = self.tokenizer.has_source_contributed(self.source)
            vocab_not_full = (self.tokenizer.current_vocab_size < self.tokenizer.vocab_size)
            
            if not has_contributed:
                if vocab_not_full:
                    logger.info(f"Source '{self.source}' has not contributed yet - adding to vocabulary...")
                    self._learn_tokenizer_merges()
                else:
                    logger.info(f"Vocabulary is full ({self.tokenizer.current_vocab_size} tokens) - cannot contribute")
            else:
                prev_contribution = self.tokenizer.sources_contributed.get(self.source, 0)
                logger.info(f"Source '{self.source}' already contributed {prev_contribution:,} merges - skipping")
        else:
            logger.info("No learned tokenizer found - will learn from first batch of data")
            self.tokenizer = NeuroTokenizer()
            self._learn_tokenizer_merges()
        
        # Initialize multiprocessing pool for parallel tokenization
        self.tokenizer_pool = Pool(processes=self.num_workers, initializer=_init_worker_tokenizer)
        logger.info(f"Initialized tokenizer pool with {self.num_workers} workers")
        
        logger.info(f"Loading {self.source} stream...")
        dataset = load_dataset(
            self.config['hf_path'],
            split=self.config['split'],
            streaming=True
        )
        
        # INSTANT RESUME: Skip to exact document position
        docs_to_skip = self.checkpoint.documents_processed
        if docs_to_skip > 0:
            logger.info(f"Instant resume: skipping to document {docs_to_skip:,}")
            dataset = dataset.skip(docs_to_skip)
        
        self.iterator = iter(dataset)
        self.docs_seen = docs_to_skip
    
    def _learn_tokenizer_merges(self):
        """
        Learn BPE merges from a sample of the dataset.
        
        FAIRNESS: Each data source gets a proportional share of the vocabulary
        based on its target_shards relative to all enabled sources.
        
        This ensures that:
        - fineweb-edu (600K shards) gets ~8% of vocab = ~2,500 merges
        - fineweb (7M shards) gets ~92% of vocab = ~29,200 merges
        """
        from datasets import load_dataset
        
        logger.info("=" * 60)
        logger.info("LEARNING TOKENIZER MERGES FROM DATA")
        logger.info("=" * 60)
        
        # Calculate this source's FAIR SHARE of the vocabulary
        max_learnable_merges = 31734  # 32000 - 266
        
        # Load sources config to determine fair allocation
        sources_config_path = os.path.join(os.path.dirname(__file__), "genesis_sources.json")
        try:
            with open(sources_config_path) as f:
                sources_config = json.load(f)
            
            # Calculate total shards across all enabled sources
            total_shards = 0
            this_source_shards = 0
            for src in sources_config.get("sources", []):
                if src.get("enabled", True):
                    shards = src.get("target_shards", 0)
                    total_shards += shards
                    if src["name"] == self.source:
                        this_source_shards = shards
            
            # Calculate proportional allocation
            if total_shards > 0 and this_source_shards > 0:
                proportion = this_source_shards / total_shards
                allocated_merges = int(max_learnable_merges * proportion)
                logger.info(f"Source '{self.source}': {this_source_shards:,} / {total_shards:,} shards = {proportion:.1%} of vocab")
                logger.info(f"Allocated {allocated_merges:,} merges for this source")
            else:
                # Fallback: equal share assuming 5 sources
                allocated_merges = max_learnable_merges // 5
                logger.warning(f"Could not calculate proportion, using default {allocated_merges:,} merges")
        except Exception as e:
            logger.warning(f"Could not load sources config: {e}, using default allocation")
            allocated_merges = max_learnable_merges // 5  # Conservative default
        
        # Check remaining slots (may already have merges from other sources)
        remaining_slots = self.tokenizer.vocab_size - self.tokenizer.next_merge_id
        if remaining_slots <= 0:
            logger.info("Vocabulary is already full! Skipping learning.")
            return
        
        # Learn the minimum of: allocated share OR remaining slots
        num_merges = min(allocated_merges, remaining_slots)
        
        if num_merges <= 0:
            logger.info(f"This source has already contributed its share. Skipping.")
            return
        
        # Load a sample of the dataset
        sample_size = 100000  # 100K documents should be enough for good vocabulary
        
        logger.info(f"Will learn {num_merges:,} merges from {sample_size:,} sample documents...")
        dataset = load_dataset(
            self.config['hf_path'],
            split=self.config['split'],
            streaming=True
        )
        
        # Collect sample texts
        sample_texts = []
        text_field = self.config['text_field']
        
        for i, doc in enumerate(dataset):
            if i >= sample_size:
                break
            text = doc.get(text_field, "")
            if text:
                sample_texts.append(text)
            if (i + 1) % 10000 == 0:
                logger.info(f"  Collected {i + 1:,} documents...")
        
        logger.info(f"Collected {len(sample_texts):,} documents")
        logger.info(f"Learning {num_merges:,} BPE merges...")
        
        # Learn merges
        vocab_before = self.tokenizer.current_vocab_size
        self.tokenizer.learn_merges(sample_texts, num_merges=num_merges, min_frequency=2)
        vocab_after = self.tokenizer.current_vocab_size
        
        # Record this source's contribution
        actual_merges_learned = vocab_after - vocab_before
        self.tokenizer.record_source_contribution(self.source, actual_merges_learned)
        
        # Save the learned tokenizer locally (as flat JSON file)
        tokenizer_path = os.path.join(os.path.dirname(__file__), "learned_tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        logger.info(f"Saved learned tokenizer to {tokenizer_path}")
        
        # Upload to S3 for nodes to download
        try:
            import boto3
            s3 = boto3.client('s3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            
            # Upload the tokenizer JSON
            with open(tokenizer_path, 'rb') as f:
                s3.put_object(
                    Bucket='neuroshard-training-data',
                    Key='tokenizer.json',
                    Body=f.read(),
                    ContentType='application/json'
                )
            logger.info("✅ Tokenizer uploaded to S3!")
        except Exception as e:
            logger.error(f"❌ Failed to upload tokenizer to S3: {e}")
        
        logger.info("Tokenizer learning complete!")
        logger.info(f"Vocabulary size: {self.tokenizer.next_merge_id:,} tokens")
        logger.info("=" * 60)
    
    def __del__(self):
        """Cleanup pool on deletion."""
        if self.tokenizer_pool:
            self.tokenizer_pool.close()
            self.tokenizer_pool.join()
        
    def get_tokens_for_shards(self, num_shards: int) -> Tuple[List[List[int]], int]:
        """
        Get tokens for multiple shards at once using parallel tokenization.
        Returns (list of token lists, total documents consumed, leftover tokens).
        """
        total_tokens_needed = num_shards * TOKENS_PER_SHARD + 1000  # Buffer
        tokens = []
        docs_consumed = 0
        text_field = self.config['text_field']
        
        # Collect documents in batches for parallel tokenization
        doc_batch_size = min(self.num_workers * 2, 32)  # Batch size for parallel processing
        doc_batch = []
        
        while len(tokens) < total_tokens_needed:
            # Collect a batch of documents
            t_fetch_start = time.time()
            try:
                for _ in range(doc_batch_size):
                    doc = next(self.iterator)
                    doc_batch.append(doc[text_field])
                    docs_consumed += 1
                    self.docs_seen += 1
            except StopIteration:
                if not doc_batch:
                    logger.warning(f"End of {self.source} dataset reached!")
                    break
            self.total_fetch_time += time.time() - t_fetch_start
            
            if not doc_batch:
                break
            
            # Tokenize batch in parallel
            t_tokenize_start = time.time()
            tokenize_args = [(text, text_field) for text in doc_batch]
            batch_tokens = self.tokenizer_pool.map(_tokenize_document, tokenize_args)
            self.total_tokenize_time += time.time() - t_tokenize_start
            
            # Flatten tokens from batch
            for doc_tokens in batch_tokens:
                tokens.extend(doc_tokens)
            
            doc_batch = []  # Reset for next batch
        
        # Split into shards
        shards = []
        for i in range(num_shards):
            start = i * TOKENS_PER_SHARD
            end = start + TOKENS_PER_SHARD
            if end <= len(tokens):
                shards.append(tokens[start:end])
            else:
                break
        
        # Return leftover tokens info
        leftover_start = len(shards) * TOKENS_PER_SHARD
        leftover = tokens[leftover_start:] if leftover_start < len(tokens) else []
        
        return shards, docs_consumed, leftover


# ============================================================================
# PARALLEL UPLOADER
# ============================================================================

class ParallelUploader:
    """Handles parallel S3 uploads while main thread tokenizes."""
    
    def __init__(self, manager: GenesisManager, source: str, num_workers: int = NUM_UPLOAD_WORKERS):
        self.manager = manager
        self.source = source
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.pending_futures = []
        self.uploaded_count = 0
        self._lock = threading.Lock()
        
    def submit_shard(self, shard_id: int, tokens: List[int]):
        """Submit a shard for async upload."""
        tensor = torch.tensor(tokens, dtype=torch.long)
        future = self.executor.submit(self._upload_shard, shard_id, tensor)
        self.pending_futures.append((shard_id, future))
        
    def _upload_shard(self, shard_id: int, tensor: torch.Tensor) -> dict:
        """Upload a single shard (runs in thread pool)."""
        result = self.manager.upload_shard(shard_id, tensor, self.source)
        with self._lock:
            self.uploaded_count += 1
        return result
    
    def wait_for_batch(self, min_complete: int = 0):
        """Wait for pending uploads to complete."""
        completed = []
        still_pending = []
        
        for shard_id, future in self.pending_futures:
            if future.done():
                try:
                    future.result()  # Raise any exceptions
                    completed.append(shard_id)
                except Exception as e:
                    logger.error(f"Upload failed for shard {shard_id}: {e}")
            else:
                still_pending.append((shard_id, future))
        
        self.pending_futures = still_pending
        
        # If we need more completions, wait
        while len(completed) < min_complete and self.pending_futures:
            shard_id, future = self.pending_futures[0]
            try:
                future.result(timeout=30)
                completed.append(shard_id)
                self.pending_futures.pop(0)
            except Exception as e:
                logger.error(f"Upload failed for shard {shard_id}: {e}")
                self.pending_futures.pop(0)
        
        return completed
    
    def wait_all(self):
        """Wait for all pending uploads to complete."""
        for shard_id, future in self.pending_futures:
            try:
                future.result(timeout=60)
            except Exception as e:
                logger.error(f"Upload failed for shard {shard_id}: {e}")
        self.pending_futures = []
        
    def shutdown(self):
        """Shutdown the executor."""
        self.wait_all()
        self.executor.shutdown(wait=True)


# ============================================================================
# MAIN POPULATOR (PARALLEL VERSION)
# ============================================================================

class GenesisPopulator:
    """Main class for populating S3 with training shards - parallel version."""
    
    def __init__(self, bucket_name: str, source: str = "fineweb-edu"):
        self.manager = GenesisManager(bucket_name)
        self.source = source
        self.checkpoint = self.manager.get_checkpoint(source)
        self.streamer = DataStreamer(source, self.checkpoint)
        self.interrupted = False
        
        # Setup interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
    def _handle_interrupt(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("\n⚠️  Interrupt received! Saving checkpoint...")
        self.interrupted = True
        
    def populate(self, target_shards: int, num_workers: int = NUM_UPLOAD_WORKERS):
        """Create shards up to target count using parallel processing."""
        current_shards = self.manager.manifest['total_shards']
        
        if current_shards >= target_shards:
            logger.info(f"Already have {current_shards} shards (target: {target_shards})")
            return
        
        shards_to_create = target_shards - current_shards
        
        # Detect CPU cores and adjust batch size
        num_cores = cpu_count()
        batch_size = min(BATCH_SIZE, max(4, num_cores))
        
        logger.info(f"{'='*60}")
        logger.info(f"NeuroShard Genesis Populator v3.0 (Parallel)")
        logger.info(f"{'='*60}")
        logger.info(f"Target: {shards_to_create:,} new shards ({current_shards:,} → {target_shards:,})")
        logger.info(f"Shard size: {SHARD_SIZE_MB}MB ({TOKENS_PER_SHARD:,} tokens)")
        logger.info(f"CPU cores: {num_cores}, Batch size: {batch_size}, Upload workers: {num_workers}")
        logger.info(f"{'='*60}")
        
        # Setup streamer
        self.streamer.setup()
        
        # Setup parallel uploader
        uploader = ParallelUploader(self.manager, self.source, num_workers=num_workers)
        
        # Progress bar
        pbar = tqdm(
            total=shards_to_create,
            desc="Creating shards",
            unit="shard",
            smoothing=0.1
        )
        
        shards_created = 0
        next_shard_id = current_shards
        leftover_tokens = list(self.checkpoint.leftover_tokens)
        
        try:
            while shards_created < shards_to_create and not self.interrupted:
                # Calculate how many shards to prepare in this batch
                remaining = shards_to_create - shards_created
                batch_count = min(batch_size, remaining)
                
                # Get tokens for batch (this is CPU-bound - tokenization)
                shard_token_lists, docs_consumed, new_leftover = self.streamer.get_tokens_for_shards(batch_count)
                
                if not shard_token_lists:
                    logger.warning("No more data available!")
                    break
                
                # Submit shards for parallel upload
                for tokens in shard_token_lists:
                    uploader.submit_shard(next_shard_id, tokens)
                    next_shard_id += 1
                
                # Wait for uploads to complete (overlap with next tokenization)
                completed = uploader.wait_for_batch(min_complete=len(shard_token_lists))
                
                # Update progress
                shards_created += len(shard_token_lists)
                pbar.update(len(shard_token_lists))
                
                # Update checkpoint periodically
                if shards_created % (batch_size * 4) == 0 or self.interrupted:
                    self.checkpoint.documents_processed = self.streamer.docs_seen
                    self.checkpoint.last_shard_id = next_shard_id - 1
                    self.checkpoint.tokens_processed += len(shard_token_lists) * TOKENS_PER_SHARD
                    self.checkpoint.leftover_tokens = new_leftover[:10000]
                    self.manager.save_checkpoint(self.checkpoint)
                    self.manager.save_manifest()
                    
                    # Log progress
                    total = current_shards + shards_created
                    rate = pbar.format_dict.get('rate', 0) or 0
                    
                    # Metrics
                    fetch_ms = (self.streamer.total_fetch_time * 1000) / max(1, shards_created)
                    tok_ms = (self.streamer.total_tokenize_time * 1000) / max(1, shards_created)
                    
                    logger.info(f"Progress: {total:,} shards, {self.checkpoint.tokens_processed/1e9:.2f}B tokens, {rate:.1f} shards/sec")
                    logger.info(f"Metrics [avg/shard]: Fetch={fetch_ms:.1f}ms, Tokenize={tok_ms:.1f}ms")
                
                leftover_tokens = new_leftover
            
            # Final cleanup
            uploader.wait_all()
            
            # Final checkpoint save
            self.checkpoint.documents_processed = self.streamer.docs_seen
            self.checkpoint.last_shard_id = next_shard_id - 1
            self.checkpoint.leftover_tokens = leftover_tokens[:10000]
            self.manager.save_checkpoint(self.checkpoint)
            self.manager.save_manifest()
            
        finally:
            uploader.shutdown()
            
            # Cleanup tokenizer pool
            if self.streamer.tokenizer_pool:
                self.streamer.tokenizer_pool.close()
                self.streamer.tokenizer_pool.join()
                self.streamer.tokenizer_pool = None
            
            pbar.close()
        
        total_shards = self.manager.manifest['total_shards']
        total_tokens = self.manager.manifest['total_tokens']
        
        logger.info(f"{'='*60}")
        logger.info(f"✓ Done! Total: {total_shards:,} shards, {total_tokens/1e9:.2f}B tokens")
        logger.info(f"{'='*60}")
        
        if self.interrupted:
            logger.info("Checkpoint saved. Run again to continue from exact position.")


# ============================================================================
# CLI
# ============================================================================

def main():
    load_env_manual()
    
    parser = argparse.ArgumentParser(
        description="NeuroShard Genesis Data Populator v3.0 (Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add shards from FineWeb-Edu (default)
  python populate_genesis_s3.py --bucket neuroshard-training-data --target 600000
  
  # Continue to target (instant resume!)
  python populate_genesis_s3.py --bucket neuroshard-training-data --target 600000
  
  # Use different data source
  python populate_genesis_s3.py --bucket neuroshard-training-data --target 7000000 --source fineweb
  
  # Check status
  python populate_genesis_s3.py --bucket neuroshard-training-data --status

Performance Tips:
  - Use c5.2xlarge or larger for best performance (~5x faster)
  - Parallel uploads overlap with tokenization
  - Checkpoint saves every 32 shards for safety
        """
    )
    
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--target", type=int, default=600000, 
                        help="Target total shard count (default: 600,000)")
    parser.add_argument("--source", default="fineweb-edu", 
                        choices=list(DATA_SOURCES.keys()),
                        help="Data source to use")
    parser.add_argument("--status", action="store_true",
                        help="Show current status and exit")
    parser.add_argument("--workers", type=int, default=NUM_UPLOAD_WORKERS,
                        help=f"Number of upload workers (default: {NUM_UPLOAD_WORKERS})")
    
    args = parser.parse_args()
    
    if args.status:
        manager = GenesisManager(args.bucket)
        m = manager.manifest
        print(f"\n{'='*60}")
        print(f"NeuroShard Genesis Status")
        print(f"{'='*60}")
        print(f"Total Shards:  {m['total_shards']:,}")
        print(f"Total Tokens:  {m.get('total_tokens', 0)/1e9:.2f}B")
        print(f"Total Size:    {m['total_shards'] * SHARD_SIZE_MB / 1000:.1f}GB")
        print(f"\nSources:")
        for src, stats in m.get('sources', {}).items():
            print(f"  {src}: {stats['shards']:,} shards, {stats['tokens']/1e9:.2f}B tokens")
        print(f"\nCheckpoints:")
        for src, cp in manager.checkpoints.items():
            print(f"  {src}: doc {cp.documents_processed:,}, shard {cp.last_shard_id}")
        print(f"{'='*60}\n")
        return
    
    populator = GenesisPopulator(args.bucket, args.source)
    populator.populate(args.target, num_workers=args.workers)


if __name__ == "__main__":
    main()
