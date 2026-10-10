#!/usr/bin/env bash
# Versioned E12 entry point. The parent starts it only after review, audit and merge.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/campaign.py" "$@"
