#!/bin/bash -e
###############################################################################
# File  :  ci/admin/install-spack.sh
###############################################################################

function cecho()
{
  local color=$1
  shift
  local text=$1
  shift
  printf "\e[${color}m${text//%/%%}\e[0m\n" >&2
}
function info() { cecho 32 "-- $@"; };
function status() { cecho "34;1" "-- $@"; };
function action() { cecho "33;1" "NOTE: $@"; };
function error() { cecho "31;1" "ERROR: $@"; exit 1; };

# Verbose call: echo before running
function vcall() { cecho "37;2" "> $*"; "$@"; }
function veval() { cecho "37;2" "> $1"; eval "$1"; }

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
info "This script is running from ${SCRIPT_DIR}"

: ${SPACK_VERSION:=develop}

###############################################################################
# Clone Spack and repositories
###############################################################################

if [ -n "${SPACK_ROOT}" ]; then
  if [ -n "${CODE}" ]; then
    : # do nothing
  elif [ -d "/rnsdhpc" ]; then
    CODE=/rnsdhpc/code
  elif [ "$(uname -s)" == "Darwin" ]; then
    CODE="/ornldev/code"
  else
    CODE=/projects
  fi

  if [ ! -e ${CODE} ]; then
    status "Creating projects directory"
    vcall sudo mkdir -p ${CODE}
    vcall sudo chown ${USER} ${CODE}
  fi

  SPACK_ROOT=${CODE}/spack
fi

if [ ! -d ${SPACK_ROOT} ]; then
  status "Installing spack"
  vcall git clone --branch develop \
    https://github.com/spack/spack.git ${SPACK_ROOT}
else
  status "Updating spack"
  veval "(cd ${SPACK_ROOT} && git fetch)"
fi

if [ "${SPACK_VERSION}" != "HEAD" ]; then
  veval "(cd ${SPACK_ROOT} && git checkout ${SPACK_VERSION})"
fi

vcall source ${SPACK_ROOT}/share/spack/setup-env.sh

###############################################################################
# Use Spack to determine platform
###############################################################################

if [ -z "${PLATFORM}" ]; then
  PLATFORM=$(spack arch --operating-system)
  info "Default PLATFORM=${PLATFORM}"
fi

if [ -z "${COMPILER}" ]; then
  case $PLATFORM in
    mojave ) COMPILER="clang@11.0.0-apple" ;;
    catalina ) COMPILER="clang@12.0.0-apple" ;;
    rhel6 ) COMPILER="gcc@4.4.7" ;;
    rhel7 ) COMPILER="gcc@4.8.5" ;;
    rhel8 ) COMPILER="gcc@8.3.1" ;;
    centos6 ) COMPILER="gcc@4.4.7" ;;
    centos7 ) COMPILER="gcc@4.8.5" ;;
    *)
      action "set \$COMPILER manually and rerun this script"
      error "No core compiler default for platform $PLATFORM" ;;
  esac
  info "Default COMPILER=${COMPILER}"
fi

###############################################################################
# Set up Spack configuration
###############################################################################

if [ ! -e ${SPACK_ROOT}/etc/spack/compilers.yaml ]; then
  vcall spack compiler find --scope site
fi

SPACK_CONFIG=${SPACK_ROOT}/etc/spack/config.yaml
if [ ! -e ${SPACK_CONFIG} ]; then
  status "Creating spack config at ${SPACK_CONFIG}"
  cat > "${SPACK_CONFIG}" << EOF
config:
  install_path_scheme: '{compiler.name}-{compiler.version}/{name}/{hash:7}'
EOF
fi

SPACK_PACKAGES="${SPACK_ROOT}/etc/spack/packages.yaml"
if [ ! -e "${SPACK_PACKAGES}" ]; then
  if [ hash nvcc 2>/dev/null ]; then
    status "Searching for external packages at ${SPACK_PACKAGES}"
    spack external find --scope=site cuda
  fi
fi

SPACK_MODULES="${SPACK_ROOT}/etc/spack/modules.yaml"
if [ ! -e "${SPACK_MODULES}" ]; then
  status "Creating default module file at ${SPACK_MODULES}"
  cat > "${SPACK_MODULES}" <<EOF
modules:
  enable::
    - lmod
  lmod:
    hierarchy:: []
    blacklist_implicits: true
    core_compilers:
      - '${COMPILER}'
    hash_length: 4
    ^python:
      autoload:  'direct'
    all:
      suffixes:
        '+cuda': 'cuda'
        '+debug': 'debug'
        '+mpi': 'mpi'
        '+multithreaded': 'mt'
        '+openmp': 'mt'
        '~shared': 'static'
        '+threads': 'mt'
        '^python@2': 'py2'
        '^zlib~shared': 'static'
        'cxxstd=98': 'c++98'
        'cxxstd=11': 'c++11'
        'cxxstd=17': 'c++17'
      environment:
        set:
          '\${PACKAGE}_ROOT': '\${PREFIX}'
        prepend_path:
          'LD_RUN_PATH': '\${PREFIX}/lib'
      filter:
        environment_blacklist: ['CPATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH',
          'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'INCLUDE']
    libffi: &lib64
      environment:
        prepend_path:
          'LD_RUN_PATH': '\${PREFIX}/lib64'
    geant4: *lib64
    jpeg-turbo: *lib64
EOF
fi

###############################################################################
# Install environments
###############################################################################

function have_spack_env() {
  cecho "37;2" "Checking for spack environment '$1'"
  test -e "$SPACK_ROOT/var/spack/environments/$1/spack.lock"
}

function update_rc() {
  local ENV=$1
  local ENVVIEW=$2
  local SENTINEL="# >>> $1 environment >>>"
  
  local BASHRC="${HOME}/.bashrc"
  local DO_UPDATE=false
  if [ ! -e "${HOME}/.bashrc" ]; then
    DO_UPDATE=true
  elif grep "${SENTINEL}" "${BASHRC}" >/dev/null; then
    DO_UPDATE=false
  elif grep "\.config/sh/rc\.sh" "${BASHRC}" >/dev/null; then
    DO_UPDATE=false
  else
    DO_UPDATE=true
  fi

  if [ "${DO_UPDATE}" == "true" ]; then
    info "Adding environment setup to ${BASHRC}"
    cat >> "${BASHRC}" <<EOF
${SENTINEL}
export CMAKE_PREFIX_PATH=${ENVVIEW}:\${CMAKE_PREFIX_PATH}
export PATH=${ENVVIEW}/bin:\${PATH}
# <<< $1 environment <<<
EOF
  fi
}

function make_spack_env() {
  local ENV=$1
  local ENVDIR=${SPACK_ROOT}/var/spack/environments/$1
  if [ -e "${ENVDIR}/spack.yaml" ]; then
    vcall rm -rf ${ENVDIR}
  fi
  local FILENAME=${2:-${SCRIPT_DIR}/env/${ENV}.yaml}
  status "Installing $ENV environment from $FILENAME"
  vcall spack env create $ENV $FILENAME
  vcall spack -e $ENV concretize
  vcall spack -e $ENV install
  local ENVVIEW=$ENVDIR/.spack-env/view
  test -d $ENVVIEW
  update_rc $ENV $ENVVIEW
}

ENV=celeritas
if ! have_spack_env $ENV ; then
  FILENAME=${SCRIPT_DIR}/env/celeritas-$(spack arch --platform).yaml
  make_spack_env $ENV $FILENAME
fi

status "Complete!"

###############################################################################
# end of ci/admin/install-spack.sh
###############################################################################
