#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Launch claude code from an apptainer image on ExCL.
# Arguments are forwarded to # the containerized 'claude' executable.
#
# Recommended use: define in your .bashrc:
# ```
#  _claude_celer() {
#    _claude_launch=$CELER_SOURCE_DIR/scripts/dev/claude-excl-celer.sh
#    echo "${_claude_launch}" "$@"
#    CELER_SOURCE_DIR=$HOME/claude/celeritas \
#    APPTAINERENV_GH_TOKEN="$(<$HOME/.config/secrets/GHAPI_SETHRJBOT_TOKEN)" \
#      "${_claude_launch}" "$@"
#  }
#  alias claude-celer=_claude_celer
# ```
#-----------------------------------------------------------------------------#

set -e

IMAGE=/auto/projects/apptainer/claude-code-notex.sif

# Persistent directories
if ! [ -d "${CELER_SOURCE_DIR}" ]; then
  echo "error: CELER_SOURCE_DIR=$CELER_SOURCE_DIR is not a directory"
  echo "(Define and use for the celeritas source)"
  exit 1
fi
CLAUDE_SRC=$CELER_SOURCE_DIR

CLAUDE_HOME="$HOME/claude/home"
CLAUDE_CCACHE="$HOME/claude/ccache"
CLAUDE_INSTALL="$HOME/claude/install"
for d in $CLAUDE_HOME $CLAUDE_CCACHE $CLAUDE_INSTALL; do
  if ! [ -d $d ]; then
    echo "Creating $d"
    mkdir -p $d
  fi
done

# Scratch directories
CLAUDE_TMP="$(mktemp -d $USER.claude.XXXXXXXX)"
if [ -z "${SCRATCHDIR}" ]; then
  SCRATCHDIR="/scratch/$USER"
fi
if ! [ -d "${SCRATCHDIR}" ]; then
  mkdir -p "$SCRATCHDIR"
  chmod 700 "$SCRATCHDIR"
fi
CLAUDE_CACHE="$SCRATCHDIR/claude/cache"
CLAUDE_BUILD="$SCRATCHDIR/claude/build"
CLAUDE_SCRATCH="$SCRATCHDIR/claude/scratch"
mkdir -p "$CLAUDE_CACHE" "$CLAUDE_BUILD" "$CLAUDE_SCRATCH"

# Environment
export APPTAINERENV_https_proxy=http://proxy.ftpn.ornl.gov:3128
export APPTAINERENV_HTTPS_PROXY=$APPTAINERENV_https_proxy
if [ -z "$APPTAINERENV_GH_TOKEN" ]; then
  echo "No APPTAINERENV_GH_TOKEN is set: claude cannot interact with github"
fi

# Helper: make a read-only bind argument
ro_path() {
  printf "%s:%s:ro" "$1" "$1"
}

# Run
exec apptainer run \
  --containall \
  --cleanenv \
  --env CELER_HOSTNAME=claude-excl \
  --env TMPDIR="/tmp" \
  --bind "$CLAUDE_TMP:/tmp" \
  --env XDG_CACHE_HOME="/cache" \
  --bind "$CLAUDE_CACHE:/cache" \
  --env CCACHE_DIR="/ccache" \
  --bind "$CLAUDE_CCACHE:/ccache" \
  --env CLAUDE_CODE_TMPDIR=/scratch \
  --bind "$CLAUDE_SCRATCH:/scratch" \
  --bind "$CLAUDE_SRC:/work/src" \
  --bind "$CLAUDE_BUILD:/work/build" \
  --bind "$CLAUDE_INSTALL:/work/install" \
  --bind $(ro_path /auto/projects/celeritas/data) \
  --bind $(ro_path /auto/projects/celeritas/opt) \
  --bind $(ro_path /auto/projects/celeritas/view) \
  --bind $(ro_path /auto/projects/celeritas/environments/scripts) \
  --bind $(ro_path /auto/software/swtree) \
  --bind $(ro_path /etc/slurm) \
  --bind $(ro_path /var/run/munge) \
  --home "$CLAUDE_HOME:/home/claude" \
  --pwd "/work" \
  "$IMAGE" "$@"
