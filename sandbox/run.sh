#!/usr/bin/env bash
# Production launch: build the frontend once, then serve everything from uvicorn.
set -euo pipefail
cd "$(dirname "$0")/.."

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

(cd webui && npm run build)
/home/stathisliap/Work/.venv/bin/uvicorn sandbox.server:app --port 8008
