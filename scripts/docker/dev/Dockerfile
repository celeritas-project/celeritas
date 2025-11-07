ARG CONFIG
FROM base-${CONFIG} as builder
LABEL maintainer="Seth Johnson <johnsonsr@ornl.gov>" \
      description="Celeritas prerequisites built with Spack"

###############################################################################
# Based on /dnf spack dockerfile:
# https://hub.docker.com/layers/spack/ubuntu-jammy/0.22
# and https://hub.docker.com/layers/spack/rockylinux9/0.22
###############################################################################

# General environment for docker
ENV DEBIAN_FRONTEND=noninteractive    \
    SPACK_ROOT=/opt/spack             \
    CURRENTLY_BUILDING_DOCKER_IMAGE=1 \
    container=docker

# Combined ubuntu/centos run command
ARG DOCKERFILE_DISTRO
RUN if [ "$DOCKERFILE_DISTRO" = "ubuntu" ] ; then \
 apt-get -yqq update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
  && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.1/ noble main" | tee --append /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/latest/ubuntu noble main" | tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update \
  && apt-get -yqq upgrade \
  && DEBIAN_FRONTEND=noninteractive apt-get -yqq install --no-install-recommends \
     build-essential ca-certificates curl file g++ gcc gfortran git gnupg2 iproute2 locales make mercurial subversion python3 python3-pip python3-setuptools unzip zstd \
     hiprand-dev rocrand-dev rocthrust-dev hipblas-dev hipcub-dev \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/* ; \
elif [ "$DOCKERFILE_DISTRO" = "rocky" ] ; then \
  dnf update -y  \
  && dnf install -y epel-release  \
  && dnf update -y  \
  && dnf --enablerepo epel install -y bzip2 curl-minimal file findutils gcc-c++ gcc gcc-gfortran git gnupg2 hg hostname iproute make patch python3 python3-pip python3-setuptools svn unzip xz zstd \
  && rm -rf /var/cache/dnf  \
  && dnf clean all ; \
fi

# XXX replaced COPY commands with this
ARG SPACK_VERSION
RUN mkdir -p $SPACK_ROOT \
  && curl -s -L https://api.github.com/repos/spack/spack/tarball/${SPACK_VERSION} \
  | tar xzC $SPACK_ROOT --strip 1

RUN ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/docker-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/interactive-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/spack-env

RUN mkdir -p /root/.spack \
 && cp $SPACK_ROOT/share/spack/docker/modules.yaml \
        /root/.spack/modules.yaml \
 && rm -rf /root/*.* /run/nologin $SPACK_ROOT/.git

WORKDIR /root
SHELL ["docker-shell"]

RUN docker-shell spack bootstrap now \
  && spack bootstrap status --optional \
  && spack spec root

ENTRYPOINT ["/bin/bash", "/opt/spack/share/spack/docker/entrypoint.bash"]
CMD ["interactive-shell"]

###############################################################################
# Install
###############################################################################

ARG CONFIG
COPY ${CONFIG}.yaml /opt/spack-environment/spack.yaml
RUN cd /opt/spack-environment && spack env activate . \
    && spack install --fail-fast

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/celeritas_spack_env.sh

# TODO: revert to default entrypoint so that commands
# (e.g. `docker run celeritas/dev:prereq bash`) work correctly
ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
