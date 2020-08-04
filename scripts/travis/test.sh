#!/bin/sh -ex
###############################################################################
# File  : scripts/travis/test.sh
###############################################################################

cd ${BUILD_DIR}

# Run tests
ctest --output-on-failure

# Run tests through valgrind
if ! ctest -D ExperimentalMemCheck --output-on-failure; then
  find Testing/Temporary -name "MemoryChecker.*.log" -exec cat {} +
  exit 1
fi

###############################################################################
# end of scripts/travis/test.sh
###############################################################################
