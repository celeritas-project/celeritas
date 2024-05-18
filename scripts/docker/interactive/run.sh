#!/bin/bash -e

if ! args=$(getopt t:m: "$@"); then
    echo "Usage: $0 [-t tag_name] [-m bind_mount]"
    exit
fi

# shellcheck disable=SC2086
set -- $args

while :; do
    case "$1" in
    -t)
      TAG_NAME=${2}
      shift; shift
      ;;
    -m)
      HOST_MOUNT=${2}
      shift; shift
      ;;
    --)
      shift; break
      ;;
    esac
done

: "${TAG_NAME:=develop}"
: "${HOST_MOUNT:=$(pwd)}"

docker run --rm -it --mount source=celeritas_storage,target=/data \
        --mount type=bind,source="${HOST_MOUNT}",target=/host \
        --cap-add=SYS_ADMIN celeritas:"${TAG_NAME}"

