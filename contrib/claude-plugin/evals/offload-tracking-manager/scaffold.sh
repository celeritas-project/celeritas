#!/bin/sh -e
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Geant4 example application with the Celeritas offload lines removed
CASE_DIR=$(cd "$(dirname "$0")" && pwd)
. "$CASE_DIR/../_lib/fixtures.sh"
copy_geant4_example_without_offload
