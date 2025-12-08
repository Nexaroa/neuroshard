#!/bin/bash
# Reset NeuroShard Ledger (preserves user accounts)
#
# This script:
# 1. Stops all containers
# 2. Removes ledger_data volume (ledger only)
# 3. Keeps postgres_data volume (user accounts)
# 4. Restarts containers with fresh ledger

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        NEUROSHARD LEDGER RESET                             ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  This will:                                                  ║"
echo "║  ✅ Reset the NEURO ledger (all balances to 0)              ║"
echo "║  ✅ Start fresh with new economics                          ║"
echo "║  ✅ Keep user accounts (postgres_data preserved)            ║"
echo "║                                                              ║"
echo "║  WARNING: This will erase all NEURO balances!               ║"
echo "║                                                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Stop containers
echo "🛑 Stopping containers..."
docker compose down

# Remove ledger volumes only
echo "🗑️  Removing ledger volumes..."
docker volume rm -f neuroshard_ledger_data 2>/dev/null || true
docker volume rm -f website_ledger_data 2>/dev/null || true

# Verify
echo ""
echo "✅ Ledger volumes removed!"
echo ""
docker volume ls | grep -E "ledger|postgres|tracker" || echo "No volumes found"
echo ""
echo "COMPLETED"

