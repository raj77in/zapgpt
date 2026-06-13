#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${CONTAINER_ENGINE:-}"

if [[ -z "$ENGINE" ]]; then
    if command -v podman >/dev/null 2>&1; then
        ENGINE="podman"
    elif command -v docker >/dev/null 2>&1; then
        ENGINE="docker"
    else
        echo "Neither podman nor docker is installed." >&2
        exit 1
    fi
fi

"$ENGINE" build -t zapgpt-test -f "$SCRIPT_DIR/Dockerfile.test" "$SCRIPT_DIR"
"$ENGINE" run --rm zapgpt-test
