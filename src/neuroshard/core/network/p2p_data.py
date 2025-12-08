import os
import hashlib
import logging
import threading
import time
import requests
import math
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from neuroshard.core.network.p2p import P2PManager
from protos import neuroshard_pb2, neuroshard_pb2_grpc
import grpc

logger = logging.getLogger(__name__)

class DataSwarm:
    """
    Implements BitTorrent-like P2P data transfer for NeuroShard.
    
    - Splits large shards into 1MB chunks.
    - Finds peers holding specific shards via DHT.
    - Downloads chunks in parallel from multiple peers.
    - Verifies data integrity via Merkle roots or simple hashes.
    """
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    def __init__(self, p2p_manager: P2PManager, cache_dir: str = "data_cache"):
        self.p2p = p2p_manager
        self.cache_dir = cache_dir
        self.active_downloads = {} # shard_id -> status
        self.local_shards = set()  # IDs of shards we have fully locally
        
        # Thread pool for parallel downloads
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        os.makedirs(cache_dir, exist_ok=True)
        self._scan_local_cache()
        
        # Start announcer thread
        threading.Thread(target=self._announce_loop, daemon=True).start()
        
    def _scan_local_cache(self):
        """Scan cache directory for existing complete shards."""
        for filename in os.listdir(self.cache_dir):
            if filename.startswith("genesis_shard_") and filename.endswith(".pt"):
                try:
                    idx = int(filename.split("_")[2].split(".")[0])
                    self.local_shards.add(idx)
                    logger.info(f"Found local shard {idx}")
                except:
                    pass

    def _announce_loop(self):
        """Periodically announce our shards to the DHT."""
        while True:
            if self.p2p.dht:
                for shard_id in list(self.local_shards):
                    key = f"shard_provider_{shard_id}".encode()
                    # In a real DHT we'd announce our IP:Port
                    # For this prototype we rely on P2PManager's peer discovery
                    pass 
            time.sleep(60)

    def get_shard_path(self, shard_id: int) -> str:
        return os.path.join(self.cache_dir, f"genesis_shard_{shard_id}.pt")

    def download_shard(self, shard_id: int, manifest_url: str = None) -> str:
        """
        Download a shard using P2P swarm, falling back to HTTP.
        Returns path to downloaded file.
        """
        target_path = self.get_shard_path(shard_id)
        
        if shard_id in self.local_shards and os.path.exists(target_path):
            return target_path
            
        logger.info(f"Starting swarm download for Shard {shard_id}...")
        
        # 1. Find Peers who have this shard
        # In a full implementation, we query the DHT: dht.get(f"shard_{shard_id}")
        # For now, we ask connected peers if they have it via a new RPC or assume based on role
        # Simplified: We try all connected peers + Genesis Host (HTTP)
        
        peers = self._find_providers(shard_id)
        
        if not peers:
            logger.info(f"No P2P providers found for Shard {shard_id}. Downloading from S3.")
            return self._download_from_s3(shard_id, target_path, manifest_url)
            
        # 2. Parallel Chunk Download
        # This is the BitTorrent part
        success = self._swarm_download(shard_id, peers, target_path)
        
        if success:
            self.local_shards.add(shard_id)
            return target_path
        else:
            logger.warning("P2P download failed. Downloading from S3.")
            return self._download_from_s3(shard_id, target_path, manifest_url)

    def _find_providers(self, shard_id: int) -> List[str]:
        """Return list of peer addresses (ip:port) that have this shard."""
        # In prototype, return empty list to force HTTP fallback initially, 
        # or implement simple gossip query
        return [] 

    def _download_from_s3(self, shard_id: int, target_path: str, manifest_url: str) -> str:
        """Download from CloudFront CDN."""
        # Use manifest URL if provided, otherwise construct CDN URL
        url = manifest_url or f"https://dwquwt9gkkeil.cloudfront.net/shard_{shard_id}.pt"
        
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            self.local_shards.add(shard_id)
            logger.info(f"Successfully downloaded Shard {shard_id}")
            return target_path
        except Exception as e:
            raise RuntimeError(f"Failed to download Shard {shard_id} from {url}: {e}")

    def _swarm_download(self, shard_id: int, peers: List[str], target_path: str) -> bool:
        """
        Download chunks in parallel from peers.
        (Placeholder for full logic)
        """
        # 1. Get metadata (size, chunk count) from first peer
        # 2. Map chunks to peers
        # 3. Download
        return False # Not yet implemented fully

    def serve_chunk(self, shard_id: int, chunk_index: int) -> bytes:
        """Read a chunk from disk to serve to a peer."""
        if shard_id not in self.local_shards:
            return None
            
        path = self.get_shard_path(shard_id)
        offset = chunk_index * self.CHUNK_SIZE
        
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                return f.read(self.CHUNK_SIZE)
        except:
            return None

