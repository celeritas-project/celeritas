FROM  celeritas:dev as builder

LABEL maintainer="Seth Johnson <johnsonsr@ornl.gov>" \
      description="Celeritas CI build"
# Run `docker build -t celeritas/ci:cuda11 .` to set up

###############################################################################
# Export environment
###############################################################################

# Remove unneeded build deps
RUN cd /opt/spack-environment && \
    spack gc -y

# Strip binaries
RUN find -L /opt/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Set up initialization
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/celeritas_spack_env.sh

###############################################################################
# Finalize
###############################################################################

# Bare OS image to run the installed executables
FROM nvidia/cuda:11.0-devel-ubuntu18.04

# Copy spack environment
COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /etc/profile.d/celeritas_spack_env.sh /etc/profile.d/celeritas_spack_env.sh

# Add core files
RUN apt-get -yqq update \
 && apt-get -yqq install --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        ssh \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 999 celeritas && \
    useradd -r -u 999 -g celeritas -d /home/celeritas -m celeritas
USER celeritas
WORKDIR /home/celeritas

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
