#------------------------------ -*- dockerfile -*- ----------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Reproduce the GitHub ubuntu-24.04 runner toolchain used by the setup-spack
# action, to build and push the Spack buildcache with
# scripts/ci/update-spack-buildcache-local.sh. Build both stages with the
# pinned Spack revisions using:
#   scripts/docker/buildcache/build.sh
# See scripts/docker/README.md for usage.
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Runner toolchain with pinned and bootstrapped Spack
#-----------------------------------------------------------------------------#
FROM ubuntu:24.04 AS spack-base

ARG SPACK_REF
ARG SPACK_PACKAGES_REF

LABEL description="Ubuntu 24.04 GitHub runner toolchain with pinned Spack" \
      org.spack.spack-ref="${SPACK_REF}" \
      org.spack.spack-packages-ref="${SPACK_PACKAGES_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# System externals must match the versions in scripts/spack/ext-ubuntu24.yaml.
# Like the GitHub runner, provide build-essential: some autotools packages
# probe for the unversioned c++ and /lib/cpp even when they only need C. The
# runner also has GCC 14, whose libstdc++ headers clang-18 selects.
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    autoconf bash build-essential clang-18 git m4 make perl \
    bzip2 ca-certificates curl file gzip libstdc++-14-dev patch python3 \
    tar unzip xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*

# Build as an unprivileged user like the GitHub runner, since some autotools
# packages refuse to configure as root. The update script installs into
# /scratch/celeritas/opt-ci when it exists: mount a volume there to keep
# installations between runs. Rootless podman (e.g. podman-hpc) must run as
# container root to access bind-mounted files owned by the host user: allow
# root to use the Spack checkouts.
RUN useradd --create-home --shell /bin/bash celeritas \
 && mkdir -p /opt/spack /opt/spack-packages /opt/spack-cache \
    /scratch/celeritas/opt-ci /work \
 && chown celeritas:celeritas /opt/spack /opt/spack-packages /opt/spack-cache \
    /scratch/celeritas/opt-ci /work \
 && git config --system --add safe.directory /opt/spack \
 && git config --system --add safe.directory /opt/spack-packages

# Match the environment of .github/actions/setup-spack. Keep the Spack user
# cache (including the bootstrap store) independent of HOME so that it is
# shared by the celeritas and root users.
ENV SPACK_ROOT=/opt/spack \
    SPACK_PACKAGES_REPO=/opt/spack-packages \
    SPACK_USER_CACHE_PATH=/opt/spack-cache \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    SPACK_COLOR=always \
    CELER_SPACK_REF=${SPACK_REF} \
    CELER_SPACK_PACKAGES_REF=${SPACK_PACKAGES_REF} \
    PATH=/opt/spack/bin:$PATH

# Check out the pinned revisions, use the pinned package repository instead
# of letting Spack clone its own, and bootstrap the concretizer
USER celeritas
RUN if [ -z "${SPACK_REF}" ] || [ -z "${SPACK_PACKAGES_REF}" ]; then \
      echo "error: SPACK_REF and SPACK_PACKAGES_REF build args are required:" \
           "use scripts/docker/buildcache/build.sh" >&2 ; \
      exit 1 ; \
    fi \
 && for pin in "${SPACK_ROOT} spack/spack ${SPACK_REF}" \
               "${SPACK_PACKAGES_REPO} spack/spack-packages ${SPACK_PACKAGES_REF}"; do \
      set -- ${pin} \
      && git init -q "$1" \
      && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD \
      || exit 1 ; \
    done \
 && spack repo set --destination "${SPACK_PACKAGES_REPO}" builtin \
 && spack bootstrap now

#-----------------------------------------------------------------------------#
# Buildcache update
#-----------------------------------------------------------------------------#
FROM spack-base AS buildcache

LABEL description="Celeritas Spack buildcache builder for GitHub CI"

COPY entrypoint.sh /opt/celeritas/

# Allow container root (see above) to configure autotools packages
ENV FORCE_UNSAFE_CONFIGURE=1 \
    CELER_SOURCE_DIR=/celeritas

USER celeritas
WORKDIR /work
ENTRYPOINT ["/opt/celeritas/entrypoint.sh"]
CMD ["/celeritas/scripts/ci/update-spack-buildcache-local.sh"]
