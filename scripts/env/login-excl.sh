#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

log error "Celeritas cannot be built from the ExCL login node"
log info "Set up your local machine's ssh configuration below"
log info "and ssh to one of the remote servers"
cat >&1 <<EOF
# .ssh/config
Host milan0 hudson faraday
  User $USER
  ProxyJump excl

Host excl
  User $USER
  HostName login.excl.ornl.gov
  IdentitiesOnly yes
  ForwardAgent yes
EOF
exit 1
