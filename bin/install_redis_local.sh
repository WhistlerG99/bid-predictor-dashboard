#!/usr/bin/env bash
# This script installs Redis and its dependencies locally into a specified directory.
# It downloads the necessary .deb packages, extracts them, and sets up environment
# variables and aliases for easy access. 
# Use this if you want to test the dashboard locally without requiring sudo 
# permissions to install redis.
set -euo pipefail

# ---- Config ----
REDIS_DIR="${REDIS_DIR:-/home/sagemaker-user/local/redis}"
BASHRC="${BASHRC:-$HOME/.bashrc}"

# ---- Packages (fixed list, no autodetection) ----
PKGS=(
  redis-server
  redis-tools
  liblzf1
  lua-cjson
  lua-bitop
  liblua5.1-0
  libjemalloc2
)

log() { echo "[$(date +'%F %T')] $*"; }

log "Installing Redis locally into: $REDIS_DIR"
mkdir -p "$REDIS_DIR"
cd "$REDIS_DIR"

log "Cleaning old downloaded .deb files (if any)..."
rm -f ./*.deb

log "Downloading packages:"
printf '  - %s\n' "${PKGS[@]}"

apt-get download "${PKGS[@]}"

log "Extracting packages into $REDIS_DIR ..."
for d in ./*.deb; do
  dpkg-deb -x "$d" .
done

# ---- Runtime env ----
LIB1="$REDIS_DIR/usr/lib/x86_64-linux-gnu"
LIB2="$REDIS_DIR/usr/lib"

export LD_LIBRARY_PATH="$LIB1:$LIB2:${LD_LIBRARY_PATH:-}"
export PATH="$REDIS_DIR/usr/bin:${PATH}"

# ---- Persist aliases + env ----
log "Writing aliases + environment to $BASHRC"

MARK_BEGIN="# >>> local redis (managed by install_redis_local.sh) >>>"
MARK_END="# <<< local redis (managed by install_redis_local.sh) <<<"

# Remove old block if present
if [[ -f "$BASHRC" ]]; then
  awk -v b="$MARK_BEGIN" -v e="$MARK_END" '
    $0==b {inblock=1; next}
    $0==e {inblock=0; next}
    !inblock {print}
  ' "$BASHRC" > "${BASHRC}.tmp"
  mv "${BASHRC}.tmp" "$BASHRC"
fi

# Add new block
{
  echo "$MARK_BEGIN"
  echo "export REDIS_DIR=\"$REDIS_DIR\""
  echo "export LD_LIBRARY_PATH=\"\$REDIS_DIR/usr/lib/x86_64-linux-gnu:\$REDIS_DIR/usr/lib:\${LD_LIBRARY_PATH:-}\""
  echo "export PATH=\"\$REDIS_DIR/usr/bin:\$PATH\""
  echo "alias redis-server=\"\$REDIS_DIR/usr/bin/redis-server\""
  echo "alias redis-cli=\"\$REDIS_DIR/usr/bin/redis-cli\""
  echo "alias redis-benchmark=\"\$REDIS_DIR/usr/bin/redis-benchmark\""
  echo "alias redis-check-aof=\"\$REDIS_DIR/usr/bin/redis-check-aof\""
  echo "alias redis-check-rdb=\"\$REDIS_DIR/usr/bin/redis-check-rdb\""
  echo "$MARK_END"
} >>"$BASHRC"

log "Done ✅"
echo
echo "Activate aliases now with:"
echo "  source \"$BASHRC\""
echo
echo "Then run:"
echo "  redis-server"
