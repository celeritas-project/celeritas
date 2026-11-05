#!/bin/bash
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Provision a Claude Code cloud VM (Ubuntu 24.04, x86_64, root) for Celeritas
# development with a spack environment identical to the main build-spack CI
# job, installed *only* from the public Celeritas spack buildcache.
#
# Usage: paste this file into the cloud environment's "Setup script" field
# (claude.ai/code: environment menu > Edit), or run it from a checkout with
#   scripts/dev/claude-cloud-setup.sh
#
# Setup takes about three minutes, under the ~5 minute limit for the
# environment snapshot to be cached and reused by later sessions. Afterward:
#   scripts/build.sh base             # configure, build, and test
#   cmake --preset=base && cmake --build --preset=base && ctest --preset=base
# using the presets in scripts/cmake-presets/claude-cloud.json .
#
# Layout (the CMake presets refer to these paths):
#   /opt/celeritas/spack           spack checkout pinned by setup-spack CI
#   /opt/celeritas/spack-packages  spack package repository pinned by CI
#   /opt/celeritas/env             spack environment (from env-ci-base.yaml)
#   /opt/celeritas/install         spack install tree
#   /opt/celeritas/view            spack view: CMAKE_PREFIX_PATH
#   /opt/celeritas/bin             cmake/ctest/ninja/ccache/spack, on PATH
#-----------------------------------------------------------------------------#

set -Eeu

CELER_OPT=/opt/celeritas

# Additional spack specs (and C++ standard) to add to env-ci-base.yaml; when
# empty, use the first "base" entry of .github/buildcache-matrix.sh, which is
# guaranteed to be in the buildcache
CELER_SPACK_SPECS=""
CELER_SPACK_CXXSTD=""

#-----------------------------------------------------------------------------#
# Helpers

SECONDS=0
log() {
  printf 'celeritas-setup [%3ds] %s: %s\n' "${SECONDS}" "$1" "$2" >&2
}
trap 'log error "failed at line ${LINENO} (exit code $?)"' ERR

# Retry a network command a few times
retry() {
  n_=1
  until "$@"; do
    if [ "${n_}" -ge 3 ]; then
      return 1
    fi
    log warning "attempt ${n_} failed: $*"
    sleep $((n_ * 5))
    n_=$((n_ + 1))
  done
}

# Find the Celeritas checkout: this script's repository, or (when pasted
# as the setup script, which starts one directory above the repositories) a
# child of the working directory
find_source_dir() {
  script_root="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd || true)"
  for d in "${script_root}" "${PWD}" "${PWD}/celeritas" "${PWD}"/*; do
    if [ -f "${d}/scripts/spack/env-ci-base.yaml" ] \
        && [ -f "${d}/.github/actions/setup-spack/action.yml" ]; then
      printf '%s\n' "${d}"
      return 0
    fi
  done
  return 1
}

# Shallow-fetch a single commit of a GitHub repository
fetch_commit() {
  repo=$1
  ref=$2
  dst=$3
  if [ "$(git -C "${dst}" rev-parse HEAD 2>/dev/null)" = "${ref}" ]; then
    log info "${repo} already at ${ref}"
    return 0
  fi
  log info "Fetching ${repo}@${ref}"
  rm -rf "${dst}"
  git init -q "${dst}"
  retry git -C "${dst}" fetch -q --depth 1 "https://github.com/${repo}.git" "${ref}"
  git -C "${dst}" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

#-----------------------------------------------------------------------------#
# Locate Celeritas

if ! CELER_SOURCE_DIR="$(find_source_dir)"; then
  log error "could not find a Celeritas checkout with scripts/spack/env-ci-base.yaml from ${PWD}"
  exit 1
fi
log info "Using Celeritas source at ${CELER_SOURCE_DIR}"

#-----------------------------------------------------------------------------#
# System packages: the buildcache uses the system clang 18 (see
# scripts/spack/ext-ubuntu24.yaml), which the cloud image already provides

missing=""
for pkg in clang-18 clang-tidy-18 g++-13 gdb git; do
  if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
    missing="${missing} ${pkg}"
  fi
done
if [ -n "${missing}" ]; then
  log info "Installing system packages:${missing}"
  export DEBIAN_FRONTEND=noninteractive
  retry apt-get update -q
  # shellcheck disable=SC2086
  retry apt-get install -y -q --no-install-recommends ${missing}
fi

#-----------------------------------------------------------------------------#
# Spack, pinned to the same commits as the setup-spack CI action

pins="$(python3 "${CELER_SOURCE_DIR}/scripts/ci/parse-spack-versions.py" \
  "${CELER_SOURCE_DIR}/.github/actions/setup-spack/action.yml")"
eval "${pins}"

mkdir -p "${CELER_OPT}"
fetch_commit "${SPACK_REPO}" "${SPACK_REF}" "${CELER_OPT}/spack"
fetch_commit "${SPACK_PACKAGES_REPO}" "${SPACK_PACKAGES_REF}" "${CELER_OPT}/spack-packages"

export SPACK_ROOT="${CELER_OPT}/spack"
export SPACK_DISABLE_LOCAL_CONFIG=1
export SPACK_COLOR=never
export PATH="${SPACK_ROOT}/bin:${PATH}"
unset SPACK_ENV

#-----------------------------------------------------------------------------#
# Spack environment, created like CI with the same base environment

if [ -z "${CELER_SPACK_SPECS}" ]; then
  # Called by the matrix file for each buildcache configuration
  build_spack_env() {
    if [ -z "${CELER_SPACK_SPECS}" ] && [ "${ENV_CI}" = "base" ]; then
      CELER_SPACK_CXXSTD="${CXXSTD}"
      CELER_SPACK_SPECS="$*"
    fi
  }
  # shellcheck source=/dev/null
  . "${CELER_SOURCE_DIR}/.github/buildcache-matrix.sh"
  unset -f build_spack_env
fi
: "${CELER_SPACK_CXXSTD:=20}"
log info "Spack environment: env-ci-base.yaml + ${CELER_SPACK_SPECS} (cxxstd=${CELER_SPACK_CXXSTD})"

envdir="${CELER_OPT}/env"
logdir="${CELER_OPT}/logs"
rm -rf "${envdir}" "${CELER_OPT}/view"
mkdir -p "${envdir}" "${logdir}"
(
  cd "${envdir}"
  # shellcheck disable=SC2086
  CXXSTD="${CELER_SPACK_CXXSTD}" \
  SPACK_ENV_FILE=env-ci-base.yaml \
  CELER_SOURCE_DIR="${CELER_SOURCE_DIR}" \
  CELER_SPACK_OPT="${CELER_OPT}/install" \
  CELER_SPACK_VIEW="${CELER_OPT}/view" \
  SPACK_PACKAGES_ROOT="${CELER_OPT}/spack-packages" \
    sh "${CELER_SOURCE_DIR}/scripts/ci/setup-spack-ci-env.sh" ${CELER_SPACK_SPECS}
)

# The CI mirror passes GITHUB_USER/GITHUB_TOKEN credentials, without which
# ghcr.io refuses the login: delete them for anonymous read-only access
sed -i '/access_pair:/,/secret_variable:/d' "${envdir}/reqs-ci.yaml"
if grep -q 'access_pair' "${envdir}/reqs-ci.yaml"; then
  log error "failed to remove buildcache credentials from ${envdir}/reqs-ci.yaml"
  exit 1
fi
# Padding is only needed for pushing to the buildcache
spack -e "${envdir}" config add "config:install_tree:padded_length:False"

log info "Bootstrapping spack"
retry spack bootstrap now >"${logdir}/bootstrap.log" 2>&1

log info "Concretizing (log: ${logdir}/concretize.log)"
if ! spack -e "${envdir}" concretize --fresh --non-defaults \
    >"${logdir}/concretize.log" 2>&1; then
  tail -n 40 "${logdir}/concretize.log" >&2
  log error "concretization failed"
  exit 1
fi

log info "Installing from buildcache (log: ${logdir}/install.log)"
if ! retry spack -e "${envdir}" install -j "$(nproc)" -p 8 \
    --use-buildcache only >"${logdir}/install.log" 2>&1; then
  grep -E 'no binary available|Error' "${logdir}/install.log" | head -n 40 >&2 || true
  log error "install failed: the buildcache may not (yet) match the CI configuration"
  exit 1
fi
spack clean --stage --downloads >/dev/null

#-----------------------------------------------------------------------------#
# Shell environment
#
# Claude Code shells keep only PATH (not other exports) from the profile, so
# the CMake presets use absolute paths and this only sets PATH: tools needed
# for building come first, everything else in the view (including its
# python3) comes *after* the system tools.

bindir="${CELER_OPT}/bin"
mkdir -p "${bindir}"
for exe in cmake ctest cpack ccmake ninja ccache; do
  ln -sf "${CELER_OPT}/view/bin/${exe}" "${bindir}/${exe}"
done
cat >"${bindir}/spack" <<EOF
#!/bin/sh
export SPACK_DISABLE_LOCAL_CONFIG=1
exec "${SPACK_ROOT}/bin/spack" "\$@"
EOF
chmod +x "${bindir}/spack"

# Full activation (e.g., for ROOT) with: . /opt/celeritas/activate.sh
spack env activate --sh -d "${envdir}" >"${CELER_OPT}/activate.sh"

path_snippet="# >>> celeritas-cloud >>>
case \":\${PATH}:\" in
  *:${bindir}:*) ;;
  *) export PATH=\"${bindir}:\${PATH}:${CELER_OPT}/view/bin\" ;;
esac
# <<< celeritas-cloud <<<"
printf '%s\n' "${path_snippet}" >/etc/profile.d/celeritas-cloud.sh
# Prepend to bashrc so that it runs before the non-interactive early return
bashrc="${HOME}/.bashrc"
if ! grep -q 'celeritas-cloud' "${bashrc}" 2>/dev/null; then
  {
    printf '%s\n' "${path_snippet}"
    if [ -f "${bashrc}" ]; then
      cat "${bashrc}"
    fi
  } >"${bashrc}.tmp"
  mv "${bashrc}.tmp" "${bashrc}"
fi

"${bindir}/ccache" --max-size=5G >/dev/null

#-----------------------------------------------------------------------------#
# Formatting and lint hooks (optional: failure does not stop the setup)

log info "Installing pre-commit hooks"
if python3 -m pip install -q --root-user-action=ignore pre-commit \
    || python3 -m pip install -q --root-user-action=ignore \
         --break-system-packages pre-commit; then
  if ! (cd "${CELER_SOURCE_DIR}" && PATH="${bindir}:${PATH}" \
          scripts/dev/install-commit-hooks.sh >"${logdir}/pre-commit.log" 2>&1); then
    log warning "failed to install pre-commit hooks: see ${logdir}/pre-commit.log"
  fi
else
  log warning "failed to install pre-commit"
fi

#-----------------------------------------------------------------------------#

log info "Geant4 $("${CELER_OPT}/view/bin/geant4-config" --version), $("${bindir}/cmake" --version | head -n 1)"
log info "Done: build with 'scripts/build.sh base' (presets: scripts/cmake-presets/claude-cloud.json)"
