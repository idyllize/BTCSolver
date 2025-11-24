#!/usr/bin/env bash
# GCE C4 Masterpiece installer (Ubuntu/Debian)
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/p71}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
USER="${USER_NAME:-p71}"
ENV_DIR="$APP_DIR/venv"
SERVICE_NAME="p71-solver"
DATA_DIR="${DATA_DIR:-/var/lib/p71}"
LOG_DIR="${LOG_DIR:-/var/log/p71}"

sudo mkdir -p "$APP_DIR" "$DATA_DIR" "$LOG_DIR"
sudo useradd -r -s /usr/sbin/nologin -d "$APP_DIR" -M "$USER" || true
sudo chown -R "$USER":"$USER" "$APP_DIR" "$DATA_DIR" "$LOG_DIR"

# Copy files
sudo cp solver.py "$APP_DIR/solver.py"
sudo cp requirements.txt "$APP_DIR/requirements.txt"
sudo chown -R "$USER":"$USER" "$APP_DIR"

# Python env
sudo -u "$USER" $PYTHON -m venv "$ENV_DIR"
sudo -u "$USER" "$ENV_DIR/bin/pip" install --upgrade pip
sudo -u "$USER" "$ENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

# Systemd unit
cat <<'UNIT' | sudo tee /etc/systemd/system/p71-solver.service >/dev/null
[Unit]
Description=Puzzle #71 Solver (GCE C4 Masterpiece)
After=network-online.target
Wants=network-online.target

[Service]
User=p71
Group=p71
Type=simple
WorkingDirectory=/opt/p71
Environment=DATA_DIR=/var/lib/p71
Environment=PROGRESS_WEBHOOK_URLS=
Environment=HIT_WEBHOOK_URLS=
Environment=REDIS_URL=
Environment=SHARD_ID=1
Environment=TOTAL_SHARDS=1
Environment=METRICS_PORT=9545
Environment=WEBHOOK_PERIOD=300
Environment=PROCESSES=
Environment=BATCH_SIZE=
ExecStart=/opt/p71/venv/bin/python /opt/p71/solver.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
AmbientCapabilities=

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable p71-solver
echo "Installation complete.

Edit /etc/systemd/system/p71-solver.service to set:
- PROGRESS_WEBHOOK_URLS (comma-separated Discord webhooks)
- HIT_WEBHOOK_URLS
- REDIS_URL (if sharding across VMs)
- SHARD_ID and TOTAL_SHARDS (e.g., 1/4, 2/4, ...)
- PROCESSES and BATCH_SIZE (optional overrides)

Then:
  sudo systemctl start p71-solver
Check logs:
  journalctl -u p71-solver -f
Metrics:
  curl http://localhost:9545/metrics
"