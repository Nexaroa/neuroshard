#!/bin/bash
# ============================================================================
# NeuroShard Genesis Controller
# ============================================================================
# Utility script to manage the Genesis data population service.
#
# Usage:
#   ./genesis_ctl.sh status    - Show current status
#   ./genesis_ctl.sh start     - Start the service
#   ./genesis_ctl.sh stop      - Stop the service
#   ./genesis_ctl.sh restart   - Restart the service
#   ./genesis_ctl.sh logs      - Tail the logs
#   ./genesis_ctl.sh install   - Install systemd service
#   ./genesis_ctl.sh add-source <name> - Add a new data source (future)
# ============================================================================

NEUROSHARD_DIR="/home/ubuntu/neuroshard"
VENV_DIR="${NEUROSHARD_DIR}/venv_build"
SERVICE_NAME="neuroshard-genesis"
BUCKET="neuroshard-training-data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          NeuroShard Genesis Data Controller                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

status() {
    print_header
    
    echo -e "${YELLOW}=== Process Status ===${NC}"
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        echo -e "${GREEN}● Genesis populator is RUNNING${NC}"
        ps aux | grep "populate_genesis_s3.py" | grep -v grep
    else
        echo -e "${RED}○ Genesis populator is NOT running${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}=== Systemd Service ===${NC}"
    if systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        echo -e "${GREEN}● Service is active${NC}"
        systemctl status ${SERVICE_NAME} --no-pager | head -10
    else
        echo -e "${RED}○ Service is not active (or not installed)${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}=== Configured Sources ===${NC}"
    if [ -f "${NEUROSHARD_DIR}/scripts/genesis_sources.json" ]; then
        python3 -c "
import json
with open('${NEUROSHARD_DIR}/scripts/genesis_sources.json') as f:
    config = json.load(f)
for s in sorted(config.get('sources', []), key=lambda x: x.get('priority', 99)):
    status = '●' if s.get('enabled', True) else '○'
    print(f\"  {status} {s['name']}: target {s.get('target_shards', 0):,} shards - {s.get('description', '')}\")
"
    else
        echo "  No config file found"
    fi
    
    echo ""
    echo -e "${YELLOW}=== S3 Data Status ===${NC}"
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate" 2>/dev/null
    
    python3 - <<'EOF'
import os
import json
import boto3

# Load env
for env_path in ['website/.env', '.env']:
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ[k] = v.strip("'").strip('"')
        break

# Load source config for targets
source_targets = {}
config_path = 'scripts/genesis_sources.json'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    for s in config.get('sources', []):
        if s.get('enabled', True):
            source_targets[s['name']] = s.get('target_shards', 500000)

try:
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )
    
    bucket = 'neuroshard-training-data'
    
    # Get manifest
    obj = s3.get_object(Bucket=bucket, Key='manifest.json')
    m = json.loads(obj['Body'].read())
    
    total_shards = m['total_shards']
    total_tokens = m.get('total_tokens', 0)
    
    # Calculate total target from all sources
    total_target = sum(source_targets.values()) if source_targets else 500000
    
    print(f"  Total Shards:    {total_shards:,} / {total_target:,} ({100*total_shards/total_target:.1f}%)")
    print(f"  Total Tokens:    {total_tokens/1e9:.2f}B")
    print(f"  Total Size:      {total_shards * 10 / 1000:.1f}GB / {total_target * 10 / 1000:.0f}GB")
    
    # Estimate remaining time based on current rate
    rate_per_hour = 3000  # ~3000 shards/hour at current speed
    remaining = total_target - total_shards
    print(f"  ETA:             ~{remaining / rate_per_hour:.0f} hours remaining")
    
    print(f"\n  Per-Source Progress:")
    for src, stats in m.get('sources', {}).items():
        target = source_targets.get(src, 500000)
        current = stats['shards']
        pct = 100 * current / target if target > 0 else 0
        bar_len = 20
        filled = int(bar_len * current / target) if target > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"    {src}:")
        print(f"      [{bar}] {pct:.1f}%")
        print(f"      {current:,} / {target:,} shards, {stats['tokens']/1e9:.2f}B tokens")
    
    # Get checkpoint
    try:
        obj = s3.get_object(Bucket=bucket, Key='checkpoints.json')
        checkpoints = json.loads(obj['Body'].read())
        print(f"\n  Active Checkpoints:")
        for src, cp in checkpoints.items():
            print(f"    {src}: doc {cp['documents_processed']:,}, last shard {cp['last_shard_id']}")
    except:
        pass
        
except Exception as e:
    print(f"  Error: {e}")
EOF
}

start() {
    print_header
    
    # Check if systemd service is installed
    if systemctl list-unit-files | grep -q ${SERVICE_NAME}; then
        echo "Starting systemd service..."
        sudo systemctl start ${SERVICE_NAME}
        sleep 2
        systemctl status ${SERVICE_NAME} --no-pager | head -5
    else
        echo "Systemd service not installed. Starting manually..."
        cd "${NEUROSHARD_DIR}"
        source "${VENV_DIR}/bin/activate"
        
        mkdir -p logs
        nohup bash scripts/genesis_service.sh > logs/genesis_service.log 2>&1 &
        
        echo -e "${GREEN}Started! PID: $!${NC}"
        echo "Logs: tail -f ${NEUROSHARD_DIR}/logs/genesis_service.log"
    fi
}

stop() {
    print_header
    
    echo "Stopping genesis populator..."
    
    # Try systemd first
    if systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        sudo systemctl stop ${SERVICE_NAME}
        echo -e "${GREEN}Systemd service stopped${NC}"
    fi
    
    # Also kill any manual processes
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        echo "Sending SIGTERM to running processes..."
        pkill -TERM -f "populate_genesis_s3.py"
        sleep 5
        
        if pgrep -f "populate_genesis_s3.py" > /dev/null; then
            echo "Processes still running, sending SIGKILL..."
            pkill -KILL -f "populate_genesis_s3.py"
        fi
    fi
    
    echo -e "${GREEN}Stopped${NC}"
}

restart() {
    stop
    sleep 2
    start
}

logs() {
    echo "Tailing genesis logs (Ctrl+C to exit)..."
    tail -f "${NEUROSHARD_DIR}/logs/genesis_service.log" "${NEUROSHARD_DIR}/logs/genesis_fineweb-edu.log" 2>/dev/null || \
    tail -f "${NEUROSHARD_DIR}/s3_population_full.log" 2>/dev/null || \
    echo "No log files found"
}

install_service() {
    print_header
    
    echo "Installing systemd service..."
    
    # Make scripts executable
    chmod +x "${NEUROSHARD_DIR}/scripts/genesis_service.sh"
    chmod +x "${NEUROSHARD_DIR}/scripts/genesis_ctl.sh"
    
    # Create logs directory
    mkdir -p "${NEUROSHARD_DIR}/logs"
    
    # Copy service file
    sudo cp "${NEUROSHARD_DIR}/scripts/neuroshard-genesis.service" /etc/systemd/system/
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable on boot
    sudo systemctl enable ${SERVICE_NAME}
    
    echo -e "${GREEN}Service installed and enabled!${NC}"
    echo ""
    echo "Commands:"
    echo "  sudo systemctl start ${SERVICE_NAME}   - Start the service"
    echo "  sudo systemctl stop ${SERVICE_NAME}    - Stop the service"
    echo "  sudo systemctl status ${SERVICE_NAME}  - Check status"
    echo "  journalctl -u ${SERVICE_NAME} -f       - View logs"
    echo ""
    echo "Or use this script:"
    echo "  ./genesis_ctl.sh start|stop|status|logs"
}

# Main
case "${1:-status}" in
    status)
        status
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    install)
        install_service
        ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|logs|install}"
        exit 1
        ;;
esac

