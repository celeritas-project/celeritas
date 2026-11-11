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
# Development image: Spack and the installed environment
#-----------------------------------------------------------------------------#
FROM ${BASE_IMAGE} AS dev

LABEL description="Celeritas GPU prerequisites built with Spack"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# Extra GPU toolkit packages (e.g. ROCm libraries used by Celeritas)
ARG GPU_APT_PACKAGES=""

# System externals must match the versions in scripts/spack/ext-ubuntu24.yaml
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    autoconf bash build-essential bzip2 ca-certificates curl file flex \
    gfortran git gzip libfl-dev m4 make patch perl python3 tar unzip \
    xz-utils zstd ${GPU_APT_PACKAGES} \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

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
RUN if [ -z "${SPACK_REF}" ] || [ -z "${SPACK_PACKAGES_REF}" ]; then \
      echo "error: SPACK_REF and SPACK_PACKAGES_REF build args are required:" \
           "use scripts/docker/build.sh" >&2 ; \
      exit 1 ; \
    fi \
 && for pin in "${SPACK_ROOT} ${SPACK_REPO} ${SPACK_REF}" \
               "${SPACK_PACKAGES_ROOT} ${SPACK_PACKAGES_REPO} ${SPACK_PACKAGES_REF}"; do \
      set -- ${pin} \
      && git init -q "$1" \
      && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD \
      || exit 1 ; \
    done \
 && spack repo set --destination "${SPACK_PACKAGES_ROOT}" builtin \
 && spack bootstrap now

# Shared Celeritas requirements and runner externals
WORKDIR /opt/spack-environment
COPY --from=spack-config \
     reqs-celer.yaml prefs-hep-nogui.yaml ext-ubuntu24.yaml ./

# GPU toolkit externals, read from the base image installation
COPY write-gpu-externals.sh /opt/docker/
RUN /opt/docker/write-gpu-externals.sh > ext-gpu.yaml \
 && cat ext-gpu.yaml

ARG CONFIG
# Extra install options, e.g. "-p 8 -j 32" for concurrent package builds
ARG SPACK_INSTALL_ARGS=""
COPY ${CONFIG}.yaml spack.yaml
RUN spack -e . concretize --fresh \
 && spack -e . install --fail-fast ${SPACK_INSTALL_ARGS}

# Modifications to the environment that are necessary to run
RUN spack env activate --sh -d . > /etc/profile.d/celeritas_spack_env.sh

COPY entrypoint.bash /opt/docker/
ENTRYPOINT ["/opt/docker/entrypoint.bash"]
CMD ["bash", "-l"]

#-----------------------------------------------------------------------------#
# Remove build-only dependencies before copying into the CI image
#-----------------------------------------------------------------------------#
FROM dev AS export

RUN spack -e /opt/spack-environment gc -y

#-----------------------------------------------------------------------------#
# CI image: base toolkit image plus the installed Spack environment
#-----------------------------------------------------------------------------#
FROM ${BASE_IMAGE} AS ci

LABEL description="Celeritas GPU CI build"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

ARG GPU_APT_PACKAGES=""

# Spack externals (compilers, perl, git) and tools for the CI actions
# (gnupg is used by the codecov uploader)
RUN apt-get -qq update \
 && apt-get -qq install -y --no-install-recommends \
    bash build-essential ca-certificates curl file git gnupg \
    libc6-dbg perl python3 python3-pip ssh vim zstd ${GPU_APT_PACKAGES} \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

COPY --from=export /opt/spack-environment /opt/spack-environment
COPY --from=export /opt/software /opt/software
COPY --from=export /opt/view /opt/view
COPY --from=export /etc/profile.d/celeritas_spack_env.sh /etc/profile.d/
COPY entrypoint.bash /opt/docker/

ENTRYPOINT ["/opt/docker/entrypoint.bash"]
CMD ["bash", "-l"]
