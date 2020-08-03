FROM  nvidia/cuda:11.0-devel-ubuntu18.04 as builder
LABEL maintainer="Seth Johnson <johnsonsr@ornl.gov>" \
      description="Celeritas prerequisites built with Spack"
# Run `docker build -t celeritas/dev:prereq .` to set up

###############################################################################
# From spack dockerfile:
# https://hub.docker.com/r/spack/ubuntu-bionic/dockerfile
# BUT replacing "COPY" commands with curl (and hard-wiring version)
###############################################################################

# General environment for docker
ENV DEBIAN_FRONTEND=noninteractive    \
    SPACK_ROOT=/opt/spack             \
    DEBIAN_FRONTEND=noninteractive    \
    CURRENTLY_BUILDING_DOCKER_IMAGE=1 \
    container=docker

RUN apt-get -yqq update \
 && apt-get -yqq install --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        file \
        g++ \
        gcc \
        gfortran \
        git \
        gnupg2 \
        iproute2 \
        lmod \
        locales \
        lua-posix \
        make \
        python3 \
        python3-pip \
        python3-setuptools \
        tcl \
        unzip \
 && locale-gen en_US.UTF-8 \
 && pip3 install boto3 \
 && rm -rf /var/lib/apt/lists/*

# XXX replaced COPY commands with this
RUN mkdir -p $SPACK_ROOT \
    && curl -s -L https://api.github.com/repos/spack/spack/tarball/v0.15.4 \
    | tar xzC $SPACK_ROOT --strip 1

RUN ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/docker-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/interactive-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/spack-env

# Add LANG default to en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN mkdir -p /root/.spack \
 && cp $SPACK_ROOT/share/spack/docker/modules.yaml \
        /root/.spack/modules.yaml \
 && rm -rf /root/*.* /run/nologin $SPACK_ROOT/.git

# [WORKAROUND]
# https://superuser.com/questions/1241548/
#     xubuntu-16-04-ttyname-failed-inappropriate-ioctl-for-device#1253889
RUN [ -f ~/.profile ]                                               \
 && sed -i 's/mesg n/( tty -s \&\& mesg n || true )/g' ~/.profile \
 || true

# [WORKAROUND]
# https://bugs.launchpad.net/ubuntu/+source/lua-posix/+bug/1752082
RUN ln -s posix_c.so /usr/lib/x86_64-linux-gnu/lua/5.2/posix.so

WORKDIR /root
SHELL ["docker-shell"]

ENTRYPOINT ["/bin/bash", "/opt/spack/share/spack/docker/entrypoint.bash"]
CMD ["interactive-shell"]

###############################################################################
# Install
###############################################################################

COPY env.yaml /opt/spack-environment/spack.yaml
RUN cd /opt/spack-environment && spack env activate . \
    && spack install --fail-fast

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/celeritas_spack_env.sh

# TODO: revert to default entrypoint so that commands
# (e.g. `docker run celeritas/dev:prereq bash`) work correctly
ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
