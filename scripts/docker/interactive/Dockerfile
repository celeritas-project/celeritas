FROM almalinux:latest

ARG PERFETTO_REP=https://android.googlesource.com/platform/external/perfetto
ARG PERFETTO_VERSION_TAG=v45.0
ENV DEVEL_ROOT=/opt/devel
ENV PERFETTO_ROOT=${DEVEL_ROOT}/perfetto
ENV PERFETTO_TOOLS=${PERFETTO_ROOT}/out/linux

# Install base utilities
RUN dnf install -y 'dnf-command(config-manager)' epel-release \
    && dnf config-manager --set-enabled crb \
    && dnf group install -y "Development Tools" \
    && dnf install -y clang clang-tools-extra ccache cmake ninja-build git sudo tmux \
      gcc-gfortran iproute python3-pip python3-boto3 python-devel vim

ENV CXX=/usr/bin/clang++

RUN mkdir -p "${DEVEL_ROOT}"

# setup perfetto
RUN git clone -b${PERFETTO_VERSION_TAG} ${PERFETTO_REP} ${PERFETTO_ROOT} \
    && cd ${PERFETTO_ROOT} \
    && tools/install-build-deps \
    && tools/gn gen --args='is_debug=false' ${PERFETTO_TOOLS} \
    && tools/ninja -C ${PERFETTO_TOOLS} tracebox traced traced_probes perfetto

# setup spack
ARG SPACK_REP=https://github.com/spack/spack.git
ARG SPACK_VERSION_TAG=v0.22.0
ARG SPACK_ENV_SPEC=${DEVEL_ROOT}/spack.yaml
ENV SPACK_ROOT=${DEVEL_ROOT}/spack
ENV SPACK_EXE=${SPACK_ROOT}/bin/spack

RUN git clone -c feature.manyFiles=true -b${SPACK_VERSION_TAG} ${SPACK_REP} ${SPACK_ROOT}

COPY spack.yaml ${SPACK_ENV_SPEC}

# install celeritas dependencies
ENV CELERITAS_ENV_NAME=celeritas

RUN ${SPACK_EXE} env create ${CELERITAS_ENV_NAME} ${SPACK_ENV_SPEC} \
    && ${SPACK_EXE} external find \
    && ${SPACK_EXE} -e ${CELERITAS_ENV_NAME} concretize \
    && ${SPACK_EXE} -e ${CELERITAS_ENV_NAME} install -j$(nproc)

# compile celeritas
ARG CELER_REP=https://github.com/celeritas-project/celeritas.git
ARG CELER_VERSION_TAG=develop
ARG CMAKE_PROFILE=reldeb-novg-profile
ENV CELER_ROOT=${DEVEL_ROOT}/celeritas

RUN git clone -b${CELER_VERSION_TAG} ${CELER_REP} ${CELER_ROOT}

COPY CMakeUserPresets.json ${CELER_ROOT}

ENV CCACHE_DIR=/data/.ccache
RUN mkdir -p ${CCACHE_DIR} /host

WORKDIR ${CELER_ROOT}

RUN eval $(${SPACK_EXE} env activate --sh ${CELERITAS_ENV_NAME}) \
    && cmake --preset=${CMAKE_PROFILE} \
    && cmake --build --preset=${CMAKE_PROFILE}

# setp entrypoint and utilities
COPY setup_shell.sh /etc/profile.d/setup_shell.sh
COPY run-perfetto.sh /usr/local/bin/run-perfetto

RUN ln -s ${SPACK_ROOT}/share/spack/docker/entrypoint.bash \
              /usr/local/bin/interactive-shell \
    && chmod u+x /usr/local/bin/run-perfetto \
    && ln -s /sys/kernel/tracing /tracing

WORKDIR /host
ENTRYPOINT ["interactive-shell"]