ARG CONFIG
FROM dev-${CONFIG} as builder

LABEL maintainer="Seth Johnson <johnsonsr@ornl.gov>" \
      description="Celeritas CI build"

###############################################################################
# Export environment
###############################################################################

# Remove unneeded build deps
RUN cd /opt/spack-environment && \
    spack gc -y

# Set up initialization
# XXX : change from >> to > since this is a duplicate
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/celeritas_spack_env.sh

###############################################################################
# Finalize
###############################################################################

# Bare OS image to run the installed executables
FROM base-${CONFIG} as parent

# Copy spack environment
COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /etc/profile.d/celeritas_spack_env.sh /etc/profile.d/celeritas_spack_env.sh

# Add core files
# TODO: gcc isn't used for rhel/centos/rocky build
ARG DOCKERFILE_DISTRO
RUN if [ "$DOCKERFILE_DISTRO" = "ubuntu" ] ; then \
  apt-get -yqq update \
  && apt-get -yqq install --no-install-recommends \
    build-essential \
    ca-certificates \
    libc6-dbg \
    g++ \
    gcc \
    ssh \
    vim \
     hiprand-dev rocrand-dev rocthrust-dev hipblas-dev hipcub-dev \
  && rm -rf /var/lib/apt/lists/* ; \
elif [ "$DOCKERFILE_DISTRO" = "rocky" ] ; then \
  /bin/sh -c dnf update -y  \
  && dnf install -y epel-release \
  && dnf update -y \
  && dnf --enablerepo epel install -y bzip2 curl-minimal file findutils gcc-c++ gcc git gnupg2 hostname iproute make patch python3 python3-pip python3-setuptools unzip vim xz zstd \
  && pip3 install boto3  && rm -rf /var/cache/dnf  && dnf clean all ; \
fi

ENTRYPOINT ["/bin/bash", "/opt/docker/entrypoint.bash"]
SHELL ["entrypoint-shell"]
CMD ["entrypoint-cmd"]
