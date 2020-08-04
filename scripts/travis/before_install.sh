#!/bin/bash -e
###############################################################################
# File  : scripts/travis/before_install.sh
###############################################################################

###############################################################################
# APT PACKAGE INSTALLS
#
# see https://docs.travis-ci.com/user/installing-dependencies
###############################################################################

set -x

if [ "${GCC_VERSION}" = "9" ]; then
  # Special repo needed for newer GCC version
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get update -q
fi

if [ -n "${GCC_VERSION}" ]; then
  # Suffix for compilers and packages
  _GCCV="-${GCC_VERSION}"

  # Download GCC packages
  sudo apt-get install gcc${_GCCV} -y
  sudo apt-get install g++${_GCCV} -y
fi

set +x

###############################################################################
# ENVIRONMENT VARIABLES
###############################################################################

export SOURCE_DIR=${PWD}
export BUILD_DIR=${SOURCE_DIR}/build
export INSTALL_DIR=${HOME}/install
export CMAKE_PREFIX_PATH=${INSTALL_DIR}:${CMAKE_PREFIX_PATH}
export PATH=${INSTALL_DIR}/bin:${PATH}
export CC=${CC:-gcc${_GCCV}}
export CXX=${CXX:-g++${_GCCV}}

mkdir -p ${BUILD_DIR}

# Define a command (using built-in travis functions) to launch one of our
# scripts
run_script() {
  set +exv
  fold_start $1 "$2"
  local scriptloc="${SOURCE_DIR}/scripts/travis/$1.sh"
  echo -e "\e[0;32mRunning ${scriptloc}\e[0m"
  ${scriptloc}
  local _RESULT=$?
  fold_end $1
  if [ ${_RESULT} -ne 0 ]; then
    echo -e "\e[1;31m${scriptloc} exited with ${_RESULT}\e[0m"
  fi
  return ${_RESULT}
}

###############################################################################
# UTILITIES FROM TRAVIS
#
# from https://github.com/travis-ci/travis-rubies/blob/build/build.sh
###############################################################################

fold_start() {
  echo -e "travis_fold:start:$1\033[33;1m$2\033[0m"
}

fold_end() {
  echo -e "\ntravis_fold:end:$1\r"
}

###############################################################################
# end of example/before_install.sh
###############################################################################
