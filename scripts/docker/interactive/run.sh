#!/bin/bash -e

# shellcheck disable=SC2048
# shellcheck disable=SC2086
if ! args=$(getopt t:m: $*); then
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

CELER_DIR=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")"/../../..)

: "${TAG_NAME:=develop}"
: "${HOST_MOUNT:=$CELER_DIR}"

docker run --rm -it --mount source=celeritas_storage,target=/data \
        --mount type=bind,source="${HOST_MOUNT}",target=/host \
        --cap-add=SYS_ADMIN celeritas:"${TAG_NAME}"

