#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run on excl (or any ubuntu24 system) to build and upload, or use the
# container in scripts/docker/buildcache (see scripts/docker/README.md)
#-----------------------------------------------------------------------------#

set -e

SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
CELER_SOURCE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Print a log message
if ! command celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi
log() {
  celerlog "$@"
}

# Ensure the git checkout at $2 matches ref $3 (description in $1)
sync_repo() {
  name=$1
  dir=$2
  ref=$3
  if [ -z "$ref" ]; then
    log error "Failed to load git ref for $name"
    exit 1
  fi
  log info "Checking $name at $dir against ref '$ref'"
  current=$(git -C "$dir" rev-parse HEAD)
  if ! wanted=$(git -C "$dir" rev-parse --verify --quiet "${ref}^{commit}"); then
    log info "Ref '$ref' not found locally: fetching"
    git -C "$dir" fetch --tags origin "$ref" >&2 || git -C "$dir" fetch --tags origin >&2
    wanted=$(git -C "$dir" rev-parse --verify "${ref}^{commit}" 2>/dev/null \
      || git -C "$dir" rev-parse --verify FETCH_HEAD)
  fi
  if [ "$current" = "$wanted" ]; then
    log info "$name already at $ref ($current)"
    return 0
  fi
  log warning "$name is at $current but expected $ref ($wanted)"
  printf "Check out %s in %s? [y/N] " "$ref" "$dir" >&2
  read -r reply
  case "$reply" in
    [yY]|[yY][eE][sS])
      git -C "$dir" checkout "$wanted" >&2
      log info "Checked out $name $ref"
      ;;
    *)
      log error "Declined to update $name"
      exit 1
      ;;
  esac
}

# Check environment
if [ -z "${SPACK_ROOT}" ]; then
  log error "SPACK_ROOT is not set: source spack's setup-env.sh first"
  exit 1
fi
log info "Using SPACK_ROOT=$SPACK_ROOT"
_pkg_loc=$("$SPACK_ROOT/bin/spack" location -P builtin)
SPACK_PACKAGES_ROOT=$(git -C "$_pkg_loc" rev-parse --show-toplevel)
log info "Using SPACK_PACKAGES_ROOT=$SPACK_PACKAGES_ROOT"

# Load spack versions
eval "$(python3 "${CELER_SOURCE_DIR}/scripts/ci/parse-spack-versions.py" \
  "${CELER_SOURCE_DIR}/.github/actions/setup-spack/action.yml")"
sync_repo "spack" "$SPACK_ROOT" "$SPACK_REF"
sync_repo "spack-packages" "$SPACK_PACKAGES_ROOT" "$SPACK_PACKAGES_REF"

# Load tokens
if [ -z "${GITHUB_USER}" ] || [ -z "${GITHUB_TOKEN}" ]; then
  _token_file="$HOME/.config/ghapi-packages-token"
  if ! [ -e "$_token_file" ]; then
    log error "Expected packages token with write privileges at $_token_file"
    exit 1
  fi
  log info "Loading tokens from $_token_file"
  . $_token_file
fi

export CELER_SOURCE_DIR GITHUB_USER GITHUB_TOKEN
exec $SCRIPT_DIR/update-spack-buildcache.sh
