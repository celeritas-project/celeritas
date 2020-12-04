#!/bin/bash

mode=exec

if [ "$(basename "$0")" == 'entrypoint-shell' ] ; then
  mode=shell
elif [ "$(basename "$0")" == 'entrypoint-cmd' ] ; then
  mode=cmd
elif [ "$1" == 'entrypoint-shell' ] ; then
  mode=shell
  shift
elif [ "$1" == 'entrypoint-cmd' ] ; then
  mode=cmd
  shift
fi

case "$mode" in
  exec)
    source /etc/profile
    exec "$@"
    ;;
  shell)
    source /etc/profile
    exec bash -c "$*"
    ;;
  cmd)
    if [ ! -t 0 ] ; then
      echo "error: no pseudo-TTY allocated: try 'docker run -it'" >&2
      exit 1
    fi
    exec bash -l
    ;;
esac
