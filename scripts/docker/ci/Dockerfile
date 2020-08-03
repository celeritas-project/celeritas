FROM  celeritas/dev:prereq as builder

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
# XXX : change from >> to > since this is a duplicate
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

# Install updated vecgeom
RUN . /etc/profile.d/celeritas_spack_env.sh \
  && cd /opt \
  && git clone --depth=1 https://gitlab.cern.ch/VecGeom/VecGeom.git vecgeom-src \
  && mkdir vecgeom-build \
  && cd vecgeom-build \
  && cmake -DBACKEND:STRING=Scalar -DBUILTIN_VECCORE:BOOL=OFF -DNO_SPECIALIZATION:BOOL=ON -DVECGEOM_VECTOR:STRING=avx2 -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_CXX_STANDARD:STRING=14 -DCUDA:BOOL=ON -DGDML:BOOL=ON -DGEANT4:BOOL=OFF -DROOT:BOOL=OFF -DBUILD_TESTING:BOOL=OFF -DCTEST:BOOL=OFF -DGDMLTESTING:BOOL=OFF -DCUDA_ARCH:STRING=70 -DCMAKE_INSTALL_PREFIX:STRING=/opt/vecgeom -G Ninja ../vecgeom-src \
  && ninja install \
  && rm -rf vecgeom-build vecgeom-src \
  && echo "export CMAKE_PREFIX_PATH=/opt/vecgeom:${CMAKE_PREFIX_PATH}" >> /etc/profile.d/celeritas_spack_env.sh

# Set up entrypoint and group
RUN groupadd -g 999 celeritas \
 && useradd -r -u 999 -g celeritas -d /home/celeritas -m celeritas \
 && ln -s /opt/docker/entrypoint.bash /usr/local/bin/entrypoint-shell \
 && ln -s /opt/docker/entrypoint.bash /usr/local/bin/entrypoint-cmd
COPY entrypoint.bash /opt/docker/entrypoint.bash

USER celeritas
WORKDIR /home/celeritas
ENTRYPOINT ["/bin/bash", "/opt/docker/entrypoint.bash"]
SHELL ["entrypoint-shell"]
CMD ["entrypoint-cmd"]
