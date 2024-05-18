#!/bin/bash

cd "${PERFETTO_ROOT}" || exit

tools/tmux -c test/configs/scheduling.cfg -C out/linux -n