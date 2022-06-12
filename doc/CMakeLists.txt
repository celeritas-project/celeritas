#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# DOXYGEN
#-----------------------------------------------------------------------------#

# File setup
set(DOXYGEN_EXCLUDE_PATTERNS
  "*.nohepmc.cc"
  "*.nompi.cc"
  "*/generated/*"
)
if(NOT CELERITAS_DEBUG)
  list(APPEND DOXYGEN_EXCLUDE_PATTERNS "*/detail/*")
endif()

set(DOXYGEN_FILE_PATTERNS
  "*.h"
  "*.hh"
  "*.cc"
  "*.cu"
  "*.md"
)
set(DOXYGEN_EXTENSION_MAPPING "cu=C++")
set(_DOXYGEN_EXTRA_SOURCE
  "${PROJECT_SOURCE_DIR}/README.md"
  "${PROJECT_SOURCE_DIR}/scripts/README.md"
  "${PROJECT_BINARY_DIR}/src/celeritas_config.h"
  "${PROJECT_BINARY_DIR}/src/celeritas_cmake_strings.h"
  "${PROJECT_BINARY_DIR}/src/celeritas_version.cc"
)

# Documentation usage
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_QT_AUTOBRIEF YES)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
set(DOXYGEN_FORMULA_MACROFILE "${CMAKE_CURRENT_SOURCE_DIR}/_static/macros.tex")
set(DOXYGEN_PREDEFINED "__DOXYGEN__")

# Verbosity/warning levels
if(NOT DEFINED DOXYGEN_QUIET)
  set(DOXYGEN_QUIET YES)
endif()
if(NOT DEFINED DOXYGEN_WARN_IF_UNDOCUMENTED)
  # Missing parameters and members have many false positives
  set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
endif()

# Output options
set(DOXYGEN_PROJECT_LOGO
  "${CMAKE_CURRENT_SOURCE_DIR}/_static/celeritas-thumbnail.png")
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_HTML_OUTPUT "doxygen-html")
set(DOXYGEN_CREATE_SUBDIRS YES)

doxygen_add_docs(doxygen
  "${PROJECT_SOURCE_DIR}/src"
  ${_DOXYGEN_EXTRA_SOURCE}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating Doxygen to doc/doxygen-html/index.html"
)

#-----------------------------------------------------------------------------#
# SPHINX/BREATHE
#-----------------------------------------------------------------------------#

if(NOT Python_FOUND)
  return()
endif()

celeritas_check_python_module(CELERITAS_USE_SPHINX sphinx)
if(NOT CELERITAS_USE_SPHINX)
  return()
endif()
celeritas_check_python_module(CELERITAS_USE_BREATHE breathe)
celeritas_check_python_module(CELERITAS_USE_SPHINXBIBTEX "sphinxcontrib.bibtex")

#-----------------------------------------------------------------------------#

function(celeritas_build_sphinx type output)
  add_custom_command(OUTPUT "${output}"
    VERBATIM COMMAND
      "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${CELERITAS_PYTHONPATH}"
        "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "$<TARGET_FILE:Python::Interpreter>" -m sphinx.cmd.build -q
        -d "${CMAKE_CURRENT_BINARY_DIR}/doctrees"
        -b ${type}
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/${type}"
    COMMENT "Building ${type} documentation with Sphinx"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ${ARGN}
  )
endfunction()

#-----------------------------------------------------------------------------#

# Define a configure file with version and configuration info
configure_file("config.json.in" "config.json" @ONLY)

set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
# Annotations break breathe parsing
foreach(_var
  CELER_FUNCTION CELER_FORCEINLINE_FUNCTION CELER_CONSTEXPR_FUNCTION
  CELER_MAYBE_UNUSED CFIF_)
  list(APPEND DOXYGEN_PREDEFINED "${_var}=")
endforeach()
# Only process top two levels (no detail, generated)
file(GLOB _DOXYGEN_SOURCE
  "${PROJECT_SOURCE_DIR}/src/*/*.hh"
  "${PROJECT_SOURCE_DIR}/src/*/*.cc"
  "${PROJECT_SOURCE_DIR}/src/*/*/*.hh"
  "${PROJECT_SOURCE_DIR}/src/*/*/*.cc"
)
doxygen_add_docs(doxygen_xml
  ${_DOXYGEN_SOURCE}
  "${PROJECT_SOURCE_DIR}/src/celeritas_version.h"
  ${_DOXYGEN_EXTRA_SOURCE}
  USE_STAMP_FILE
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating Doxygen XML"
)

set(_doc_html "${CMAKE_CURRENT_BINARY_DIR}/html/index.html")
celeritas_build_sphinx(html "${_doc_html}")
add_custom_target(doc DEPENDS "${_doc_html}")
if(CELERITAS_USE_BREATHE)
  add_dependencies(doc doxygen_xml)
endif()

#-----------------------------------------------------------------------------#
