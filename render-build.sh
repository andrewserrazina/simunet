#!/usr/bin/env bash
set -euo pipefail

# Disable Docker BuildKit because Render's environment might not have a buildkitd
# daemon available for rootless builds. Falling back to the classic builder keeps
# deployments working without requiring Docker BuildKit support.
export DOCKER_BUILDKIT=0

echo "Render build script executed: DOCKER_BUILDKIT disabled for compatibility."
