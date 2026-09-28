#------------------------------ -*- dockerfile -*- ----------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Reproduce the GitHub ubuntu-24.04 runner toolchain used by the setup-spack
# action, to build and push the Spack buildcache with
# scripts/ci/update-spack-buildcache-local.sh. Build from the repository root:
#   docker build -f scripts/docker/buildcache/Dockerfile -t celeritas-buildcache .
# See scripts/docker/README.md for usage.
#-----------------------------------------------------------------------------#

FROM ubuntu:24.04

LABEL description="Celeritas Spack buildcache builder for GitHub CI"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# System externals must match the versions in scripts/spack/ext-ubuntu24.yaml.
# The GitHub runner also has GCC 14, whose libstdc++ headers clang-18 selects.
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    autoconf bash clang-18 g++-13 gcc-13 gfortran-13 git m4 make perl \
    bzip2 ca-certificates curl file gzip libstdc++-14-dev patch python3 \
    python3-yaml tar unzip xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*

# Build as an unprivileged user like the GitHub runner, since some autotools
# packages refuse to configure as root. The update script installs into
# /scratch/celeritas/opt-ci when it exists: mount a volume there to keep
# installations between runs.
RUN useradd --create-home --shell /bin/bash celeritas \
 && mkdir -p /opt/spack /opt/spack-packages /scratch/celeritas/opt-ci /work \
 && chown celeritas:celeritas \
    /opt/spack /opt/spack-packages /scratch/celeritas/opt-ci /work

# Match the environment of .github/actions/setup-spack
ENV SPACK_ROOT=/opt/spack \
    SPACK_PACKAGES_REPO=/opt/spack-packages \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    SPACK_COLOR=always \
    CELER_SOURCE_DIR=/celeritas \
    PATH=/opt/spack/bin:$PATH

COPY scripts/docker/buildcache/entrypoint.sh /opt/celeritas/entrypoint.sh
COPY .github/actions/setup-spack/action.yml \
     /opt/celeritas/src/.github/actions/setup-spack/action.yml

# Check out the currently pinned Spack revisions and bootstrap the
# concretizer; the entrypoint updates the checkouts at run time if the pins
# in the mounted source differ
USER celeritas
RUN CELER_SOURCE_DIR=/opt/celeritas/src /opt/celeritas/entrypoint.sh \
    spack bootstrap now

WORKDIR /work
ENTRYPOINT ["/opt/celeritas/entrypoint.sh"]
CMD ["/celeritas/scripts/ci/update-spack-buildcache-local.sh"]
