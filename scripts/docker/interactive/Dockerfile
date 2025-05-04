FROM almalinux:latest

ARG PERFETTO_REP=https://android.googlesource.com/platform/external/perfetto
ARG PERFETTO_VERSION_TAG=v49.0
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
    && tools/gn gen --args='is_debug=false is_hermetic_clang=false' ${PERFETTO_TOOLS} \
    && tools/ninja -C ${PERFETTO_TOOLS} tracebox traced traced_probes perfetto

# setup spack
ARG SPACK_REP=https://github.com/spack/spack.git
ARG SPACK_VERSION_TAG=develop
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
ARG CMAKE_PROFILE=reldeb-profile
ENV CELER_ROOT=${DEVEL_ROOT}/celeritas

RUN git clone -b${CELER_VERSION_TAG} ${CELER_REP} ${CELER_ROOT}

COPY CMakeUserPresets.json ${CELER_ROOT}

ENV CCACHE_DIR=/data/.ccache
RUN mkdir -p ${CCACHE_DIR} /host

WORKDIR ${CELER_ROOT}

RUN eval $(${SPACK_EXE} env activate --sh ${CELERITAS_ENV_NAME}) \
    && cmake --preset=${CMAKE_PROFILE} \
    && cmake --build --preset=${CMAKE_PROFILE} \
    && cmake --install build-${CMAKE_PROFILE}

# compile geomodel
ENV GEOMODEL_ROOT=${DEVEL_ROOT}/GeoModel
RUN dnf install -y eigen3-devel hdf5-devel sqlite-devel \
    && git clone -bceleritas https://github.com/celeritas-project/GeoModel.git ${GEOMODEL_ROOT} \
    && eval $(${SPACK_EXE} env activate --sh ${CELERITAS_ENV_NAME}) \
    && mkdir -p ${GEOMODEL_ROOT}/build && cd ${GEOMODEL_ROOT}/build \
    && git submodule update --init \
    && cmake -DGEOMODEL_BUILD_ATLASEXTENSIONS=ON -DGEOMODEL_BUILD_FULLSIMLIGHT=ON -DGEOMODEL_USE_HEPMC3=ON .. \
    && make -j$(nproc)

ENV LANG=en_US.UTF-8

# setup entrypoint and utilities
COPY setup_shell.sh /etc/profile.d/setup_shell.sh
COPY run-perfetto.sh /usr/local/bin/run-perfetto
COPY scheduling.cfg ${PERFETTO_ROOT}/celer.cfg

RUN ln -s ${SPACK_ROOT}/share/spack/docker/entrypoint.bash \
              /usr/local/bin/interactive-shell \
    && chmod u+x /usr/local/bin/run-perfetto \
    && ln -s /sys/kernel/tracing /tracing

WORKDIR /host
ENTRYPOINT ["interactive-shell"]