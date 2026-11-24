#------------------------------ -*- dockerfile -*- ----------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# CUDA/ROCm development and CI images with Celeritas dependencies installed by
# the Spack revisions pinned for CI. Build both stages with:
#   scripts/docker/build.sh {cuda|rocm}
# See scripts/docker/README.md for usage.
#-----------------------------------------------------------------------------#

# Base GPU toolkit image (nvidia/cuda devel or rocm/dev)
ARG BASE_IMAGE

#-----------------------------------------------------------------------------#
# Metadata, environment, and system packages shared by both images. Changes to
# this stage rebuild the Spack environment: add CI tools to the CI stage.
#-----------------------------------------------------------------------------#
FROM ${BASE_IMAGE} AS base

# See https://github.com/opencontainers/image-spec/blob/main/annotations.md
# The build script adds the labels that change with each build (created,
# revision, version) so that they do not invalidate cached layers.
ARG BASE_IMAGE
LABEL org.opencontainers.image.authors="Celeritas contributors" \
      maintainer="Celeritas contributors" \
      org.opencontainers.image.vendor="Celeritas Project" \
      org.opencontainers.image.url="https://celeritas-project.github.io/celeritas/" \
      org.opencontainers.image.source="https://github.com/celeritas-project/celeritas" \
      org.opencontainers.image.documentation="https://github.com/celeritas-project/celeritas/blob/develop/scripts/docker/README.md" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# Extra GPU toolkit packages (e.g. ROCm libraries used by Celeritas)
ARG GPU_APT_PACKAGES=""

# Compilers and Spack externals needed to build both the environment and
# Celeritas
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    bash build-essential ca-certificates curl file git perl python3 zstd \
    ${GPU_APT_PACKAGES} \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

#-----------------------------------------------------------------------------#
# Development image: Spack and the installed environment
#-----------------------------------------------------------------------------#
FROM base AS dev

LABEL org.opencontainers.image.title="Celeritas GPU development image" \
      org.opencontainers.image.description="Celeritas GPU prerequisites built with Spack"

# Build-only packages for Spack. System externals, including those installed
# in the base stage, must match the versions in scripts/spack/ext-ubuntu24.yaml
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    autoconf bzip2 flex gfortran gzip libfl-dev m4 make patch tar unzip \
    xz-utils \
 && rm -rf /var/lib/apt/lists/*

ARG SPACK_REPO
ARG SPACK_REF
ARG SPACK_PACKAGES_REPO
ARG SPACK_PACKAGES_REF

LABEL org.spack.spack-ref="${SPACK_REF}" \
      org.spack.spack-packages-ref="${SPACK_PACKAGES_REF}"

ENV SPACK_ROOT=/opt/spack \
    SPACK_PACKAGES_ROOT=/opt/spack-packages \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    SPACK_COLOR=always \
    PATH=/opt/spack/bin:$PATH

# Check out the pinned revisions, use the pinned package repository instead
# of letting Spack clone its own, and bootstrap the concretizer
RUN <<'EOF'
set -e
if [ -z "${SPACK_REF}" ] || [ -z "${SPACK_PACKAGES_REF}" ]; then
  echo "error: SPACK_REF and SPACK_PACKAGES_REF build args are required:" \
       "use scripts/docker/build.sh" >&2
  exit 1
fi
checkout() {
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3"
  git -C "$1" checkout -q --detach FETCH_HEAD
}
checkout "${SPACK_ROOT}" "${SPACK_REPO}" "${SPACK_REF}"
checkout "${SPACK_PACKAGES_ROOT}" "${SPACK_PACKAGES_REPO}" "${SPACK_PACKAGES_REF}"
spack repo set --destination "${SPACK_PACKAGES_ROOT}" builtin
spack bootstrap now
EOF

# Environment and shared configuration from scripts/spack
WORKDIR /opt/spack-environment
ARG CONFIG
COPY --from=spack-config \
     reqs-celer.yaml reqs-docker.yaml prefs-build.yaml prefs-hep-nogui.yaml \
     ext-ubuntu24.yaml ./
COPY --from=spack-config env-docker-${CONFIG}.yaml spack.yaml

# GPU toolkit externals, read from the base image installation
COPY write-gpu-externals.sh /opt/docker/
RUN /opt/docker/write-gpu-externals.sh > ext-gpu.yaml \
 && cat ext-gpu.yaml

# Extra install options, e.g. "-p 8 -j 32" for concurrent package builds
ARG SPACK_INSTALL_ARGS=""
RUN spack -e . concretize --fresh \
 && spack -e . install --fail-fast ${SPACK_INSTALL_ARGS}

# Modifications to the environment that are necessary to run
RUN spack env activate --sh -d . > /etc/profile.d/celeritas_spack_env.sh

COPY entrypoint.bash /opt/docker/
ENTRYPOINT ["/opt/docker/entrypoint.bash"]
CMD ["bash", "-l"]

#-----------------------------------------------------------------------------#
# Remove build-only dependencies and Spack metadata (the build logs and specs
# of each installation, and the database) before copying into the CI image
#-----------------------------------------------------------------------------#
FROM dev AS export

RUN spack -e /opt/spack-environment gc -y \
 && find /opt/software /opt/view/ -name '.spack*' -prune -exec rm -rf {} +

#-----------------------------------------------------------------------------#
# CI image: base stage plus the installed Spack environment
#-----------------------------------------------------------------------------#
FROM base AS ci

LABEL org.opencontainers.image.title="Celeritas GPU CI image" \
      org.opencontainers.image.description="GPU toolkit and Celeritas prerequisites for CI builds"

# Tools for the CI actions (gnupg is used by the codecov uploader)
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    gnupg libc6-dbg python3-pip ssh vim \
 && rm -rf /var/lib/apt/lists/*

COPY --from=export /opt/spack-environment /opt/spack-environment
COPY --from=export /opt/software /opt/software
COPY --from=export /opt/view /opt/view
COPY --from=export /etc/profile.d/celeritas_spack_env.sh /etc/profile.d/
COPY entrypoint.bash /opt/docker/

ENTRYPOINT ["/opt/docker/entrypoint.bash"]
CMD ["bash", "-l"]
