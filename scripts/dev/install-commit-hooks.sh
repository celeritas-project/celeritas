#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! hash pre-commit 2>/dev/null ; then
  printf "\e[31mpre-commit is not installed.\e[0m
Install pre-commit and update your PATH.
" >&2
  for pgm in brew pip yum dnf; do
    if hash ${pgm} 2>/dev/null ; then
      printf "\e[33;40mmaybe:\e[0m ${pgm} install pre-commit\n" >&2
    fi
  done
  printf "See \e[34;40mhttps://pre-commit.com\e[0m\n" >&2
  exit 1
fi

HOOK_DIR="$(git rev-parse --git-common-dir)/hooks"
for script in pre-commit post-commit; do
  FILENAME="${HOOK_DIR}/${script}"
  if grep -l "scripts/dev" "${FILENAME}" >/dev/null 2>&1; then
    printf "\e[31mDisabling obsolete ${script} hook at ${FILENAME}.\e[0m\n" >&2
    mv "$FILENAME" "$FILENAME.disabled"
  fi
done

pre-commit install --install-hooks
