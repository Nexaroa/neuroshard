#!/usr/bin/env python3
"""
NeuroShard CLI - Main entry point for running a node

Usage:
    neuroshard-node --port 8000 --token YOUR_TOKEN
    neuroshard-node --daemon --token YOUR_TOKEN
    neuroshard-node --stop
    neuroshard-node --help

This starts a NeuroShard node that:
1. Participates in distributed LLM training
2. Earns NEURO tokens via Proof of Neural Work
3. Serves a web dashboard at http://localhost:PORT/
"""

import argparse
import sys
import os
import webbrowser
import threading
import time
import signal

from neuroshard.version import __version__

# Paths for daemon mode
NEUROSHARD_DIR = os.path.expanduser("~/.neuroshard")
PID_FILE = os.path.join(NEUROSHARD_DIR, "node.pid")
LOG_FILE = os.path.join(NEUROSHARD_DIR, "node.log")


def open_dashboard_delayed(port: int, delay: float = 3.0):
    """Open the dashboard in browser after a delay (to let server start)."""
    def opener():
        time.sleep(delay)
        url = f"http://localhost:{port}/"
        print(f"\n[NODE] Opening dashboard: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[NODE] Could not open browser: {e}")
            print(f"[NODE] Please manually open: {url}")
    
    thread = threading.Thread(target=opener, daemon=True)
    thread.start()


def ensure_neuroshard_dir():
    """Ensure ~/.neuroshard directory exists."""
    os.makedirs(NEUROSHARD_DIR, exist_ok=True)


def get_running_pid():
    """Get PID of running daemon, or None if not running."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        # Check if process is actually running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file exists but process is dead
        try:
            os.remove(PID_FILE)
        except:
            pass
        return None


def stop_daemon():
    """Stop the running daemon."""
    pid = get_running_pid()
    if pid is None:
        print("[NODE] No running daemon found")
        return False
    
    print(f"[NODE] Stopping daemon (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to stop
        for _ in range(30):  # 3 seconds max
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            # Force kill if still running
            print("[NODE] Sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)
        
        # Clean up PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("[NODE] ✓ Daemon stopped")
        return True
    except Exception as e:
        print(f"[NODE] Error stopping daemon: {e}")
        return False


def show_status():
    """Show daemon status."""
    pid = get_running_pid()
    if pid:
        print(f"[NODE] ✓ Daemon running (PID {pid})")
        print(f"[NODE]   Dashboard: http://localhost:8000/")
        print(f"[NODE]   Logs: {LOG_FILE}")
        return True
    else:
        print("[NODE] ✗ Daemon not running")
        return False


def tail_logs(lines: int = 50):
    """Show recent log entries."""
    if not os.path.exists(LOG_FILE):
        print(f"[NODE] No log file found at {LOG_FILE}")
        return
    
    print(f"[NODE] Last {lines} lines of {LOG_FILE}:")
    print("-" * 60)
    try:
        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                print(line, end='')
    except Exception as e:
        print(f"[NODE] Error reading logs: {e}")


def daemonize(port: int):
    """Fork process to run as daemon."""
    ensure_neuroshard_dir()
    
    # Check if already running
    existing_pid = get_running_pid()
    if existing_pid:
        print(f"[NODE] Daemon already running (PID {existing_pid})")
        print(f"[NODE] Use 'neuroshard --stop' to stop it first")
        sys.exit(1)
    
    print(f"[NODE] Starting daemon...")
    print(f"[NODE]   Logs: {LOG_FILE}")
    print(f"[NODE]   PID file: {PID_FILE}")
    print(f"[NODE]   Dashboard: http://localhost:{port}/")
    
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent exits
            print(f"[NODE] ✓ Daemon started (PID {pid})")
            sys.exit(0)
    except OSError as e:
        print(f"[NODE] Fork failed: {e}")
        sys.exit(1)
    
    # Decouple from parent
    os.chdir("/")
    os.setsid()
    os.umask(0)
    
    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.exit(1)
    
    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Open log file
    log_fd = open(LOG_FILE, 'a')
    
    # Redirect stdout/stderr to log file
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())
    
    # Close stdin
    devnull = open('/dev/null', 'r')
    os.dup2(devnull.fileno(), sys.stdin.fileno())
    
    # Write PID file
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # Register cleanup on exit
    import atexit
    def cleanup():
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except:
                pass
    atexit.register(cleanup)
    
    # Log startup
    print(f"\n{'='*60}")
    print(f"[DAEMON] NeuroShard daemon started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DAEMON] PID: {os.getpid()}")
    print(f"[DAEMON] Port: {port}")
    print(f"{'='*60}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroShard Node - Decentralized AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a node (foreground)
  neuroshard --token YOUR_TOKEN

  # Start as background daemon
  neuroshard --daemon --token YOUR_TOKEN

  # Stop the daemon
  neuroshard --stop

  # Check daemon status
  neuroshard --status

  # View logs
  neuroshard --logs

  # Run on custom port
  neuroshard --port 9000 --token YOUR_TOKEN

  # Inference-only mode
  neuroshard --token YOUR_TOKEN --no-training

Get your wallet token at: https://neuroshard.com/wallet
        """
    )
    
    # Core options
    parser.add_argument(
        "--port", type=int, default=8000,
        help="HTTP port for the node (default: 8000)"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="Wallet token (64-char hex) or 12-word mnemonic phrase"
    )
    parser.add_argument(
        "--tracker", type=str, default="https://neuroshard.com/api/tracker",
        help="Tracker URL for peer discovery (bootstrap only)"
    )
    parser.add_argument(
        "--seed-peers", type=str, default=None,
        help="Comma-separated seed peers for DHT bootstrap (e.g., '1.2.3.4:8000,5.6.7.8:8001')"
    )
    
    # Network options
    parser.add_argument(
        "--announce-ip", type=str, default=None,
        help="Force this IP address for peer announcements"
    )
    parser.add_argument(
        "--announce-port", type=int, default=None,
        help="Force this port for peer announcements"
    )
    
    # Training options
    parser.add_argument(
        "--no-training", action="store_true",
        help="Disable training (inference only)"
    )
    parser.add_argument(
        "--observer", action="store_true",
        help="Observer mode: sync ledger from network but don't generate proofs (for explorer)"
    )
    parser.add_argument(
        "--diloco-steps", type=int, default=500,
        help="DiLoCo inner steps before gradient sync (default: 500)"
    )
    
    # Device options
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device: auto (default), cuda, mps, or cpu"
    )
    
    # Resource limits
    parser.add_argument(
        "--memory", type=int, default=None,
        help="Max memory in MB (default: auto-detect 70%% of system RAM)"
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=None,
        help="Max CPU threads to use (default: all cores)"
    )
    parser.add_argument(
        "--max-storage", type=int, default=100,
        help="Max disk space for training data in MB (default: 100)"
    )
    
    # UI options
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open dashboard in browser"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Don't auto-open browser (same as --no-browser)"
    )
    
    # Daemon options
    parser.add_argument(
        "--daemon", "-d", action="store_true",
        help="Run as background daemon (logs to ~/.neuroshard/node.log)"
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop the running daemon"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check if daemon is running"
    )
    parser.add_argument(
        "--logs", action="store_true",
        help="Show recent daemon logs"
    )
    parser.add_argument(
        "--log-lines", type=int, default=50,
        help="Number of log lines to show with --logs (default: 50)"
    )
    
    # Info options
    parser.add_argument(
        "--version", action="version",
        version=f"NeuroShard {__version__}"
    )
    
    args = parser.parse_args()
    
    # Handle daemon control commands first (before requiring token)
    if args.stop:
        success = stop_daemon()
        sys.exit(0 if success else 1)
    
    if args.status:
        success = show_status()
        sys.exit(0 if success else 1)
    
    if args.logs:
        tail_logs(args.log_lines)
        sys.exit(0)
    
    # Detect GPU before printing banner
    gpu_status = "CPU"
    gpu_color = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_status = f"CUDA ({gpu_name})"
            gpu_color = "\033[92m"  # Green
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_status = "Apple Metal (MPS)"
            gpu_color = "\033[92m"  # Green
        else:
            gpu_status = "CPU (no GPU detected)"
            gpu_color = "\033[93m"  # Yellow
    except ImportError:
        gpu_status = "PyTorch not installed"
        gpu_color = "\033[91m"  # Red
    
    reset_color = "\033[0m"
    
    # Print banner
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗               ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗              ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║              ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║              ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝              ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝              ║
║                                                              ║
║   ███████╗██╗  ██╗ █████╗ ██████╗ ██████╗                   ║
║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗                  ║
║   ███████╗███████║███████║██████╔╝██║  ██║                  ║
║   ╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║                  ║
║   ███████║██║  ██║██║  ██║██║  ██║██████╔╝                  ║
║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝                  ║
║                                                              ║
║            Decentralized AI Training Network                 ║
║                     v{__version__:<10}                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Print GPU status
    print(f"  {gpu_color}🖥️  Device: {gpu_status}{reset_color}")
    print()
    
    # Validate token
    if not args.token:
        print("[ERROR] Wallet token required!")
        print()
        print("Get your token at: https://neuroshard.com/wallet")
        print()
        print("Usage: neuroshard --token YOUR_TOKEN")
        print("       neuroshard --daemon --token YOUR_TOKEN")
        sys.exit(1)
    
    # Daemonize if requested (must happen after banner so user sees feedback)
    if args.daemon:
        # Check platform - daemon mode only works on Unix
        if sys.platform == 'win32':
            print("[ERROR] Daemon mode not supported on Windows")
            print("[NODE] Use Windows Task Scheduler or run in foreground")
            sys.exit(1)
        daemonize(args.port)
    
    # Auto-open browser (unless disabled or daemon mode)
    if not args.no_browser and not args.headless and not args.daemon:
        open_dashboard_delayed(args.port)
    
    # Import runner from the package
    from neuroshard.runner import run_node
    
    # Handle mnemonic input
    node_token = args.token
    if node_token:
        words = node_token.strip().split()
        if len(words) == 12:
            try:
                from mnemonic import Mnemonic
                mnemo = Mnemonic("english")
                if mnemo.check(node_token):
                    seed = mnemo.to_seed(node_token, passphrase="")
                    node_token = seed[:32].hex()
                    print("[NODE] ✅ Wallet recovered from mnemonic")
                else:
                    print("[WARNING] Invalid mnemonic - treating as raw token")
            except ImportError:
                print("[WARNING] 'mnemonic' package not installed")
            except Exception as e:
                print(f"[WARNING] Mnemonic error: {e}")
    
    # Run the node
    print(f"[NODE] Starting on port {args.port}...")
    print(f"[NODE] Dashboard: http://localhost:{args.port}/")
    print()
    
    # Parse seed peers
    seed_peers = None
    if args.seed_peers:
        seed_peers = [p.strip() for p in args.seed_peers.split(',') if p.strip()]
    
    run_node(
        port=args.port,
        tracker=args.tracker,
        node_token=node_token,
        announce_ip=args.announce_ip,
        announce_port=args.announce_port,
        enable_training=not args.no_training,
        observer_mode=args.observer,
        available_memory_mb=args.memory,
        max_storage_mb=args.max_storage,
        max_cpu_threads=args.cpu_threads,
        diloco_inner_steps=args.diloco_steps,
        device=args.device,
        seed_peers=seed_peers,
    )


if __name__ == "__main__":
    main()
