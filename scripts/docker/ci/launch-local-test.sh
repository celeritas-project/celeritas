#!/bin/sh -e

if [ -z "$1" ]; then
  echo "Usage: CONFIG= BUILD= $0 pull-request-id" 1>&2
  exit 2;
fi

if [ -z "${CONFIG}" ]; then
  CONFIG=jammy-cuda11
  echo "Set default CONFIG=${CONFIG}" 1>&2
fi
if [ -z "${BUILD}" ]; then
  BUILD=full-novg
  echo "Set default BUILD=${BUILD}" 1>&2
fi

case ${CONFIG} in
  focal-cuda* ) OS=ubuntu-cuda ;;
  jammy-cuda* ) OS=ubuntu-cuda ;;
  bionic-minimal ) OS=ubuntu-minimal ;;
  centos*-rocm* ) OS=centos-rocm ;;
  * )
    echo "Unknown CONFIG='${CONFIG}'" 1>&2
    exit 1
    ;;
esac

DOCKER=docker
if ! hash ${DOCKER} 2>/dev/null; then
  DOCKER=podman
  if ! hash ${DOCKER} 2>/dev/null; then
    echo "Docker (or podman) is not available" 1>&2
    exit 1
  fi
fi


CONTAINER=$(${DOCKER} run -t -d ci-${CONFIG})
echo "Launched container: ${CONTAINER}" 1>&2
${DOCKER} exec -i ${CONTAINER} bash -l <<EOF || echo "*BUILD FAILED*"
set -e
git clone https://github.com/celeritas-project/celeritas.git src
cd src
git fetch origin pull/$1/head:mr/$1
git checkout mr/$1
entrypoint-shell ./scripts/ci/run-ci.sh ${OS} ${BUILD}
EOF
${DOCKER} stop --time=0 ${CONTAINER}
echo "To resume: ${DOCKER} start ${CONTAINER} \\" 1>&2
echo "           && ${DOCKER} exec -it -e 'TERM=xterm-256color' ${CONTAINER} bash -l" 1>&2
echo "To delete: ${DOCKER} rm -f $CONTAINER" 1>&2
