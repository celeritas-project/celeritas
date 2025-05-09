#!/bin/sh

usage() { echo "Usage: $0 [-t tag_name] [-r repo_url]"; exit 1; }

while getopts "t:r:" opt; do
    case "${opt}" in
    t)
      TAG_NAME=${OPTARG}
      ;;
    r)
      REPO_URL=${OPTARG}
      ;;
    ?)
      usage
      ;;
    esac
done
shift $((OPTIND-1))

set -e

: "${TAG_NAME:=develop}"
: "${REPO_URL:=https://github.com/celeritas-project/celeritas.git}"

SCRIPT_DIR=$(dirname "$0")

docker build -t celeritas:"${TAG_NAME}" --build-arg CELER_REP="${REPO_URL}" \
       --build-arg CELER_VERSION_TAG="${TAG_NAME}" "${SCRIPT_DIR}"
