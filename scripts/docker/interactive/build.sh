#!/bin/bash

if ! args=$(getopt t:r: "$@"); then
    echo "Usage: $0 [-t tag_name] [-r repo_url]"
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
    -r)
      REPO_URL=${2}
      shift; shift
      ;;
    --)
      shift; break
      ;;
    esac
done

: "${TAG_NAME:=develop}"
: "${REPO_URL:=https://github.com/celeritas-project/celeritas.git}"

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

echo $TAG_NAME
echo $REPO_URL

docker build -t celeritas:"${TAG_NAME}" --build-arg CELER_REP="${REPO_URL}" \
       --build-arg CELER_VERSION_TAG="${TAG_NAME}" "${SCRIPT_DIR}"