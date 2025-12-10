import grpc
import time
from typing import Dict
import threading

class ConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConnectionPool, cls).__new__(cls)
                    cls._instance.channels = {} # url -> channel
                    cls._instance.last_used = {} # url -> timestamp
        return cls._instance

    def get_channel(self, address: str):
        """
        Get an existing channel or create a new one.
        address format: "ip:port"
        """
        # Normalize address (remove http:// if present)
        if address.startswith("http://"):
            address = address.replace("http://", "")
        if address.startswith("https://"):
            address = address.replace("https://", "")

        with self._lock:
            if address in self.channels:
                # Check if channel is active (simplified check)
                self.last_used[address] = time.time()
                return self.channels[address]
            
            # Create new channel for P2P network
            # Fast keepalive to detect dead nodes quickly in decentralized network
            # IMPORTANT: Increase message size for activation tensors in pipeline training!
            MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB for large batches/sequences
            options = [
                ('grpc.keepalive_time_ms', 30000),  # Ping every 30 seconds
                ('grpc.keepalive_timeout_ms', 10000),  # 10 second timeout
                ('grpc.keepalive_permit_without_calls', True),  # Ping even when idle
                ('grpc.http2.max_pings_without_data', 0),  # Unlimited pings
                ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),  # For receiving responses
                ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),  # For sending activations
            ]
            channel = grpc.insecure_channel(address, options=options)
            self.channels[address] = channel
            self.last_used[address] = time.time()
            return channel

    def cleanup(self, max_idle_seconds=300):
        """Close channels idle for too long"""
        now = time.time()
        to_remove = []
        with self._lock:
            for addr, last_time in self.last_used.items():
                if now - last_time > max_idle_seconds:
                    to_remove.append(addr)
            
            for addr in to_remove:
                print(f"Closing idle connection to {addr}")
                self.channels[addr].close()
                del self.channels[addr]
                del self.last_used[addr]

# Global accessor
def get_channel(address: str):
    pool = ConnectionPool()
    return pool.get_channel(address)

