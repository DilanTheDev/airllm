#!/usr/bin/env bash
# Builds dist/AirLLM.exe (portable, no installer). Runs on Linux, macOS or Windows (Git Bash):
# Wails cross-compiles the Windows binary without cgo.
#
# Needs: Go 1.22+, Node 18+, and the Wails CLI:
#   go install github.com/wailsapp/wails/v2/cmd/wails@v2.12.0
set -euo pipefail
cd "$(dirname "$0")"

# 1. Bundle the worker and this repo's airllm package into the exe.
rm -rf bundle
mkdir -p bundle
cp worker/airllm_worker.py bundle/
cp -r ../air_llm/airllm bundle/airllm
find bundle -name '__pycache__' -type d -prune -exec rm -rf {} +
find bundle -name '*.pyc' -delete

# 2. Icon.
go run ./tools/genicon build

# 3. Build. Wails runs the frontend build from wails.json.
WAILS="${WAILS:-$(command -v wails || echo "$(go env GOPATH)/bin/wails")}"
"$WAILS" build -platform windows/amd64 -skipbindings -trimpath -ldflags "-s -w" -clean -o AirLLM.exe

mkdir -p dist
cp build/bin/AirLLM.exe dist/AirLLM.exe
echo "built $(pwd)/dist/AirLLM.exe"
