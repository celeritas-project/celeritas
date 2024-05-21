#!/bin/bash

cd "${PERFETTO_ROOT}" || exit

tools/tmux -c celer.cfg -C out/linux -n