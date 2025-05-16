#!/bin/sh -e

cd "${PERFETTO_ROOT}" || exit

exec tools/tmux -c celer.cfg -C out/linux -n
