#!/bin/sh -e
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Fake Celeritas install whose GPU architectures do not match the device: a
# CUDA build for compute capability 70 running on a capability 80 device. Its
# celer-config mirrors the corecel/io/BuildOutput.cc layout. Do not describe
# the problem in comments in that file, since the run can read it.
CASE_DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p fake-install/bin
cp "$CASE_DIR/fake-celer-config" fake-install/bin/celer-config
chmod +x fake-install/bin/celer-config
