#!/bin/bash
# Run the Evolve-Anything Streamlit Viewer
# Usage: ./run_viewer.sh [port] [search_root]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8501}"
SEARCH_ROOT="${2:-$(pwd)}"

echo "🧬 Starting Evolve-Anything Viewer..."
echo "📁 Search root: $SEARCH_ROOT"
echo "🌐 Port: $PORT"
echo ""

cd "$SEARCH_ROOT"
streamlit run "$SCRIPT_DIR/streamlit_app.py" --server.port "$PORT"
