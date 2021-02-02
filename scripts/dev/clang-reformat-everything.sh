#!/bin/sh -ex
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

GIT_WORK_TREE="$(git rev-parse --show-toplevel)"
find $GIT_WORK_TREE \( -name '*.hh' -or -name '*.cc' -or -name '*.cu' \) \
  -exec clang-format -i {} +
