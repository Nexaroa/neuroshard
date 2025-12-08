#!/usr/bin/env python3
"""
Extract Node Credentials from Token

This script shows you the derived wallet address and private key
for a given node token, so you can back them up securely.

Uses proper ECDSA cryptography (secp256k1) - same as Bitcoin/Ethereum!
"""

import hashlib
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual crypto module used by NeuroShard
try:
    from neuroshard.core.crypto import derive_keypair_from_token
    ECDSA_AVAILABLE = True
except ImportError:
    ECDSA_AVAILABLE = False
    print("⚠️  WARNING: ECDSA crypto not available. Install: pip install cryptography")
    print()


def extract_credentials(node_token: str):
    """Extract wallet address and private key from node token using ECDSA."""
    
    if ECDSA_AVAILABLE:
        # Use the REAL crypto system (ECDSA with secp256k1)
        keypair = derive_keypair_from_token(node_token)
        
        private_key_hex = keypair.private_key_bytes.hex()
        public_key_hex = keypair.public_key_bytes.hex()
        node_id = keypair.node_id
        
        # Wallet ID is first 16 chars of node_id
        wallet_id = node_id[:16]
        
        # Wallet address is the full node_id (derived from public key hash)
        wallet_address = node_id
        
        return {
            "node_token": node_token,
            "private_key": private_key_hex,
            "public_key": public_key_hex,
            "node_id": node_id,
            "wallet_id": wallet_id,
            "wallet_address": wallet_address,
            "crypto_type": "ECDSA (secp256k1)",
        }
    else:
        # Fallback (INSECURE - just for display, not for actual use)
        private_key = hashlib.sha256(node_token.encode()).hexdigest()
        node_id = str(int(hashlib.sha256(node_token.encode()).hexdigest(), 16))
        wallet_id = hashlib.sha256(node_token.encode()).hexdigest()[:16]
        wallet_address = hashlib.sha256(node_token.encode()).hexdigest()
        
        return {
            "node_token": node_token,
            "private_key": private_key,
            "public_key": "N/A (ECDSA not available)",
            "node_id": node_id[:32] + "...",
            "wallet_id": wallet_id,
            "wallet_address": wallet_address,
            "crypto_type": "INSECURE FALLBACK (install cryptography package!)",
        }


def main():
    # Check if token provided as argument
    if len(sys.argv) > 1:
        node_token = sys.argv[1]
    else:
        # Default: observer token from docker-compose.yml
        node_token = "observer_node_token_secure_123"
        print(f"Using default observer token from docker-compose.yml")
        print()
    
    credentials = extract_credentials(node_token)
    
    print("=" * 70)
    print("NEUROSHARD NODE CREDENTIALS")
    print("=" * 70)
    print()
    print(f"Crypto Type:     {credentials['crypto_type']}")
    print()
    print("⚠️  KEEP THESE SECRET - They control access to your NEURO wallet!")
    print()
    print(f"Node Token:      {credentials['node_token']}")
    print(f"Private Key:     {credentials['private_key']}")
    print(f"Public Key:      {credentials['public_key']}")
    print()
    print(f"Node ID:         {credentials['node_id']}")
    print(f"Wallet Address:  {credentials['wallet_address']}")
    print(f"Wallet ID:       {credentials['wallet_id']}")
    print()
    print("=" * 70)
    print(".ENV FORMAT (copy this to your .env file):")
    print("=" * 70)
    print()
    print(f"# Observer Node Credentials (ECDSA-based)")
    print(f"OBSERVER_NODE_TOKEN={credentials['node_token']}")
    print(f"OBSERVER_PRIVATE_KEY={credentials['private_key']}")
    print(f"OBSERVER_PUBLIC_KEY={credentials['public_key']}")
    print(f"OBSERVER_WALLET_ADDRESS={credentials['wallet_address']}")
    print(f"OBSERVER_NODE_ID={credentials['node_id']}")
    print(f"OBSERVER_WALLET_ID={credentials['wallet_id']}")
    print()
    print("=" * 70)
    print("SECURITY NOTES:")
    print("=" * 70)
    print()
    print("🔒 KEEP SECRET (Never Share):")
    print("   • Node Token - Master credential that derives everything")
    print("   • Private Key - Used to sign PoNW proofs")
    print()
    print("🌍 SAFE TO SHARE (Public):")
    print("   • Public Key - Anyone can use this to verify your signatures")
    print("   • Wallet Address/Node ID - Your public NEURO address")
    print("   • Wallet ID - Short identifier for display")
    print()
    print("⚠️  CRITICAL: Anyone with the token or private key can:")
    print("   • Sign PoNW proofs as you")
    print("   • Steal your NEURO rewards")
    print("   • Impersonate your node")
    print()
    print("✅ The public key and wallet address are SAFE to share publicly.")
    print("   They're used by other nodes to verify your PoNW proofs.")
    print()
    print("📝 Recommended: Add .env to .gitignore to prevent accidental commits")
    print()


if __name__ == "__main__":
    main()

