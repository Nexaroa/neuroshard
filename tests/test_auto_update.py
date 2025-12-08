#!/usr/bin/env python3
"""
Test script for auto-update functionality.

Run this to verify the update detection and download URL generation.
"""

import platform
import sys


def test_version_comparison():
    """Test semantic version comparison logic."""
    print("=" * 60)
    print("Testing Version Comparison")
    print("=" * 60)
    
    def compare(v1, v2):
        """Returns True if v1 > v2."""
        try:
            t1 = tuple(int(x) for x in v1.split('.')[:3])
            t2 = tuple(int(x) for x in v2.split('.')[:3])
            return t1 > t2
        except:
            return False
    
    test_cases = [
        ("0.4.2", "0.4.1", True),   # Patch update
        ("0.5.0", "0.4.9", True),   # Minor update
        ("1.0.0", "0.9.9", True),   # Major update
        ("0.4.1", "0.4.1", False),  # Same version
        ("0.4.0", "0.4.1", False),  # Older version
    ]
    
    for v1, v2, expected in test_cases:
        result = compare(v1, v2)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {v1} > {v2}: {result} (expected {expected})")
    
    print()


def test_platform_detection():
    """Test platform and variant detection."""
    print("=" * 60)
    print("Testing Platform Detection")
    print("=" * 60)
    
    system = platform.system()
    print(f"  Platform: {system}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python version: {sys.version}")
    
    # Try to detect GPU support
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f"  PyTorch CUDA available: {has_cuda}")
        print(f"  PyTorch MPS available: {has_mps}")
        
        if has_cuda:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("  PyTorch not installed (will use CPU build)")
    
    print()


def test_download_url_generation():
    """Test download URL generation for different platforms."""
    print("=" * 60)
    print("Testing Download URL Generation")
    print("=" * 60)
    
    version = "v0.4.2"
    base_url = f"https://d1qsvy9420pqcs.cloudfront.net/releases/{version}"
    
    system = platform.system()
    
    if system == "Windows":
        # Detect CPU vs GPU
        try:
            import torch
            if torch.cuda.is_available():
                url = f"{base_url}/NeuroShardNode-GPU.exe"
                filename = "NeuroShardNode-GPU.exe"
                variant = "GPU (CUDA detected)"
            else:
                url = f"{base_url}/NeuroShardNode-CPU.exe"
                filename = "NeuroShardNode-CPU.exe"
                variant = "CPU (no CUDA)"
        except ImportError:
            url = f"{base_url}/NeuroShardNode-CPU.exe"
            filename = "NeuroShardNode-CPU.exe"
            variant = "CPU (PyTorch not installed)"
        
        print(f"  Platform: Windows ({variant})")
        print(f"  Download URL: {url}")
        print(f"  Filename: {filename}")
    
    elif system == "Darwin":
        url = f"{base_url}/NeuroShardNode_Mac.zip"
        filename = "NeuroShardNode_Mac.zip"
        arch = platform.machine()
        variant = "Apple Silicon" if arch == "arm64" else "Intel"
        
        print(f"  Platform: macOS ({variant})")
        print(f"  Download URL: {url}")
        print(f"  Filename: {filename}")
    
    elif system == "Linux":
        print(f"  Platform: Linux")
        print(f"  Auto-update not supported (use package manager or build from source)")
        print(f"  Manual download: https://neuroshard.com/download")
    
    else:
        print(f"  Platform: {system} (unsupported)")
    
    print()


def test_api_endpoint():
    """Test connection to update manifest endpoint."""
    print("=" * 60)
    print("Testing Update Manifest API")
    print("=" * 60)
    
    import requests
    
    url = "https://neuroshard.com/api/downloads/latest"
    
    try:
        print(f"  Fetching: {url}")
        resp = requests.get(url, timeout=5)
        
        if resp.status_code == 200:
            print(f"  ✅ Status: {resp.status_code} OK")
            data = resp.json()
            
            print(f"\n  Response:")
            print(f"    Tag: {data.get('tag_name', 'N/A')}")
            print(f"    Name: {data.get('name', 'N/A')}")
            print(f"    Published: {data.get('published_at', 'N/A')}")
            
            # Extract version
            latest = data.get("tag_name", "").lstrip('v')
            print(f"\n  Latest version: {latest}")
            
        else:
            print(f"  ❌ Status: {resp.status_code}")
            print(f"  Response: {resp.text[:200]}")
    
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
        print(f"\n  Make sure the API endpoint is deployed!")
        print(f"  Expected endpoint: {url}")
    
    print()


def test_download_availability():
    """Test if CloudFront CDN has the builds available."""
    print("=" * 60)
    print("Testing CloudFront Build Availability")
    print("=" * 60)
    
    import requests
    
    # Test latest/ builds
    builds = [
        "https://d1qsvy9420pqcs.cloudfront.net/releases/latest/NeuroShardNode-CPU.exe",
        "https://d1qsvy9420pqcs.cloudfront.net/releases/latest/NeuroShardNode-GPU.exe",
        "https://d1qsvy9420pqcs.cloudfront.net/releases/latest/NeuroShardNode_Mac.zip",
    ]
    
    for url in builds:
        filename = url.split('/')[-1]
        try:
            # HEAD request to check existence without downloading
            resp = requests.head(url, timeout=5, allow_redirects=True)
            
            if resp.status_code == 200:
                size_mb = int(resp.headers.get('Content-Length', 0)) / (1024 * 1024)
                print(f"  ✅ {filename}")
                print(f"     Size: {size_mb:.1f} MB")
                print(f"     URL: {url}")
            else:
                print(f"  ❌ {filename} - Status: {resp.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {filename} - Error: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NeuroShard Auto-Update Test Suite")
    print("=" * 60)
    print()
    
    test_version_comparison()
    test_platform_detection()
    test_download_url_generation()
    test_api_endpoint()
    test_download_availability()
    
    print("=" * 60)
    print("Test suite complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

