#!/bin/sh

usage() { echo "Usage: $0 [-t tag_name] [-m bind_mount]"; exit 1; }

while getopts "t:m:" opt; do
    case "${opt}" in
    t)
      TAG_NAME=${OPTARG}
      ;;
    m)
      HOST_MOUNT=${OPTARG}
      ;;
    ?)
      usage
      ;;
    esac
done
shift $((OPTIND-1))

set -e

CELER_DIR=$(readlink -f "$(dirname "$0")"/../../..)

: "${TAG_NAME:=develop}"
: "${HOST_MOUNT:=$CELER_DIR}"


exec docker run --rm -it --mount source=celeritas_storage,target=/data \
        --mount type=bind,source="${HOST_MOUNT}",target=/host \
        --cap-add=SYS_ADMIN celeritas:"${TAG_NAME}"
