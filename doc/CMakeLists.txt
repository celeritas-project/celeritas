#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# DOXYGEN
#-----------------------------------------------------------------------------#

# File setup
set(DOXYGEN_EXCLUDE_PATTERNS
  "*.nocuda.hh"
  "*.nocuda.cc"
  "*.nohepmc.cc"
  "*.nompi.cc"
  "*.nompi.i.hh"
  "*.noroot.cc"
)
if(NOT CELERITAS_DEBUG)
  list(APPEND DOXYGEN_EXCLUDE_PATTERNS "*/detail/*")
endif()

set(DOXYGEN_FILE_PATTERNS
  "*.h"
  "*.hh"
  "*.cc"
  "*.cu"
)
set(DOXYGEN_EXTENSION_MAPPING "cu=C++")

# Documentation usage
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_QT_AUTOBRIEF YES)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)

# Verbosity/warning levels
if(NOT DEFINED DOXYGEN_QUIET)
  set(DOXYGEN_QUIET YES)
endif()
if(NOT DEFINED DOXYGEN_WARN_IF_UNDOCUMENTED)
  # Missing parameters and members have many false positives
  set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
endif()

# Output options
set(DOXYGEN_GENERATE_XML ${_USE_BREATHE})
set(DOXYGEN_PROJECT_LOGO
  "${CMAKE_CURRENT_SOURCE_DIR}/_static/celeritas-logo.png")

doxygen_add_docs(doc
  "${PROJECT_SOURCE_DIR}/src"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Processing C++ documentation with Doxygen"
)

#-----------------------------------------------------------------------------#
