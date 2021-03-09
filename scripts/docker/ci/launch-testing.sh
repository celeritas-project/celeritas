#!/bin/sh -e

if [ -z "$1" ]; then
  echo "Usage: $0 pull-request-id" 1>&2
  exit 2;
fi


if [ -z "${CONFIG}" ]; then
  echo "Defaulting CONFIG=bionic-minimal"
  CONFIG=focal-cuda11
fi

CONTAINER=$(docker run -t -d ci-${CONFIG})
echo "Launched container: ${CONTAINER}"
docker exec -i $CONTAINER bash -l <<EOF || echo "*BUILD FAILED*"
set -e
git clone https://github.com/celeritas-project/celeritas src
cd src
git fetch origin pull/$1/head:mr/$1
git checkout mr/$1
SOURCE_DIR=. BUILD_DIR=build entrypoint-shell ./scripts/build/docker.sh
EOF
docker stop --time=0 $CONTAINER
echo "To resume: docker start $CONTAINER \\"
echo "           && docker exec -it -e 'TERM=xterm-256color' $CONTAINER bash -l"
echo "To delete: docker rm -f $CONTAINER"
