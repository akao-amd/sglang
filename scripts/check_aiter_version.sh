#!/bin/bash
#
# Check if AITER wheel needs to be rebuilt based on docker/rocm.Dockerfile
#
# Returns:
#   - Sets REBUILD_NEEDED=true if rebuild needed
#   - Sets REBUILD_NEEDED=false if existing wheel can be reused
#   - Sets AITER_VERSION to the detected version
#
# Exit codes:
#   0 - Success (outputs can be parsed)
#   1 - Error (missing Dockerfile, AWS credentials, etc.)

set -e

ROCM_VERSIONS=("700" "720")
DOCKERFILE="docker/rocm.Dockerfile"

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Error: $DOCKERFILE not found" >&2
  exit 1
fi

# Extract AITER_COMMIT from Dockerfile
AITER_COMMIT=$(grep -E '^\s*ARG\s+AITER_COMMIT=' "$DOCKERFILE" | head -1 | sed 's/.*AITER_COMMIT=\s*//; s/["'\'']//g' | tr -d ' ')

if [[ -z "$AITER_COMMIT" ]]; then
  echo "Error: AITER_COMMIT not found in $DOCKERFILE" >&2
  exit 1
fi

echo "Found AITER_COMMIT: $AITER_COMMIT" >&2

# Determine AITER version format
if [[ $AITER_COMMIT =~ ^v[0-9] ]]; then
  # Version tag (e.g., v0.1.12.post1)
  AITER_VERSION="$AITER_COMMIT"
  echo "AITER_COMMIT is a version tag: $AITER_VERSION" >&2
else
  # Commit SHA - need to trace git history to find version
  echo "AITER_COMMIT is a SHA, tracing git history..." >&2

  # Clone AITER repo temporarily if not already cloned
  AITER_REPO="https://github.com/ROCm/aiter.git"
  TEMP_DIR=$(mktemp -d)

  git clone --quiet "$AITER_REPO" "$TEMP_DIR/aiter" >&2 || {
    echo "Error: Failed to clone AITER repository" >&2
    rm -rf "$TEMP_DIR"
    exit 1
  }

  cd "$TEMP_DIR/aiter"

  # Try to find nearest version tag for this commit
  AITER_VERSION=$(git describe --tags "$AITER_COMMIT" 2>/dev/null || echo "unknown")

  cd - > /dev/null
  rm -rf "$TEMP_DIR"

  if [[ "$AITER_VERSION" == "unknown" ]]; then
    echo "Warning: Could not determine version for commit $AITER_COMMIT" >&2
    echo "Will use commit SHA as version" >&2
    AITER_VERSION="$AITER_COMMIT"
  else
    echo "Traced commit $AITER_COMMIT to version: $AITER_VERSION" >&2
  fi
fi

# Check S3 for existing wheels
REBUILD_NEEDED=false
S3_BUCKET="${AMD_S3_BUCKET_NAME:-aioss-pypi-prod}"

echo "Checking S3 bucket: $S3_BUCKET" >&2

for ROCM_VER in "${ROCM_VERSIONS[@]}"; do
  S3_PATH="s3://${S3_BUCKET}/sglang/rocm${ROCM_VER}/packages/aiter/"

  echo "Checking: $S3_PATH" >&2

  # List wheels in S3 (requires AWS credentials)
  if ! WHEELS=$(aws s3 ls "$S3_PATH" 2>/dev/null); then
    echo "Warning: Could not list S3 path $S3_PATH (AWS credentials or path may not exist)" >&2
    # If we can't list S3, assume rebuild is needed
    REBUILD_NEEDED=true
    break
  fi

  # Check if this AITER version exists
  # Expected pattern: aiter-{VERSION}+rocm{ROCM_VER}-*.whl
  # Version might be like 0.1.12 or v0.1.12.post1
  VERSION_PATTERN="${AITER_VERSION#v}"  # Remove 'v' prefix if present

  if ! echo "$WHEELS" | grep -q "aiter-${VERSION_PATTERN}+rocm${ROCM_VER}"; then
    echo "AITER version $AITER_VERSION not found for rocm${ROCM_VER}" >&2
    REBUILD_NEEDED=true
    break
  else
    echo "Found existing AITER wheel for rocm${ROCM_VER}" >&2
  fi
done

# Output results (can be parsed by GitHub Actions)
echo "REBUILD_NEEDED=$REBUILD_NEEDED"
echo "AITER_VERSION=$AITER_VERSION"
echo "AITER_COMMIT=$AITER_COMMIT"

exit 0
