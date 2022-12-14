#!/bin/sh -e

if [ -z "$1" ]; then
  echo "Usage: $0 pull-request-id" 1>&2
  exit 2;
fi

if [ -z "${CONFIG}" ]; then
  CONFIG=focal-cuda11
  echo "Set default CONFIG=${CONFIG}"
fi
if [ -z "${BUILD}" ]; then
  BUILD=full-novg
  echo "Set default BUILD=${BUILD}"
fi

DOCKER=docker
if ! hash ${DOCKER} 2>/dev/null; then
  DOCKER=podman
  if ! hash ${DOCKER} 2>/dev/null; then
    echo "Docker (or podman) is not available"
    exit 1
  fi
fi


CONTAINER=$(${DOCKER} run -t -d ci-${CONFIG})
echo "Launched container: ${CONTAINER}"
${DOCKER} exec -i ${CONTAINER} bash -l <<EOF || echo "*BUILD FAILED*"
set -e
git clone https://github.com/celeritas-project/celeritas.git src
cd src
git fetch origin pull/$1/head:mr/$1
git checkout mr/$1
entrypoint-shell ./scripts/ci/run-ci.sh ${BUILD}
EOF
${DOCKER} stop --time=0 ${CONTAINER}
echo "To resume: ${DOCKER} start ${CONTAINER} \\"
echo "           && ${DOCKER} exec -it -e 'TERM=xterm-256color' ${CONTAINER} bash -l"
echo "To delete: ${DOCKER} rm -f $CONTAINER"
