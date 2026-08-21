#!/usr/bin/env bash
# Build (or live-rebuild) Z-mag docs and serve HTML over HTTP.
#
# Usage:
#   ./serve.sh              # live rebuild + serve on 127.0.0.1:8765
#   DOCS_PORT=9000 ./serve.sh
#
# If the server runs on a remote host, forward the port:
#   ssh -L 8765:127.0.0.1:8765 USER@HOST
#   open http://127.0.0.1:8765/

set -euo pipefail
cd "$(dirname "$0")"

PORT="${DOCS_PORT:-8765}"
HOST="${DOCS_HOST:-127.0.0.1}"

if ! command -v dot >/dev/null 2>&1; then
    echo "Warning: Graphviz 'dot' not found; some flowchart figures may be skipped." >&2
fi

if [[ "${DOCS_ONCE:-0}" == "1" ]]; then
    make html
    echo "Built static HTML in build/html/"
    exit 0
fi

if command -v sphinx-autobuild >/dev/null 2>&1; then
    echo "Live docs: http://${HOST}:${PORT}/  (Ctrl+C to stop)"
    echo "  algorithms: http://${HOST}:${PORT}/theory_algorithms.html"
    exec sphinx-autobuild source build/html \
        --host "$HOST" \
        --port "$PORT" \
        --watch ../Zmag_dynamic_stereo_base.py \
        --watch ../Zmag_dynamic_stereo_demag.py \
        --watch ../Zmag_dynamic_stereo_nodemag.py
fi

echo "sphinx-autobuild not found; run: pip install -r requirements.txt" >&2
make html
echo "Serving http://${HOST}:${PORT}/  (Ctrl+C to stop)"
cd build/html
exec python -m http.server "$PORT" --bind "$HOST"
