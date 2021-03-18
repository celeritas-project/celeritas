#!/bin/sh -e
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

GIT_WORK_TREE="$(git rev-parse --show-toplevel)"
if ! git diff-files --quiet ; then
  printf "\e[1;33mWill not git-clang-format: repository is dirty\e[0m. \e[33m \
Stage (with 'git add') or stash (with 'git stash') all files before committing \
to apply formatting.\e[0m
" >&2
  exit 1
fi
  
set -x
find $GIT_WORK_TREE \( -name '*.hh' -or -name '*.cc' -or -name '*.cu' \) \
  -exec clang-format -i {} +

SKIP_GCF=1 git commit -a -m "Format code base ($(clang-format --version))" \
  --author "Clang-format <celeritas-project@users.noreply.github.com>"
