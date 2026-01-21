#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_version_to_hex(_doxy_version_hex DOXYGEN_VERSION)

#-----------------------------------------------------------------------------#
# DEVELOPER MANUAL (Doxygen)
#-----------------------------------------------------------------------------#

set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)

# Use full version string instead of short version
set(DOXYGEN_PROJECT_NUMBER "${Celeritas_VERSION_STRING}")

# File setup
set(DOXYGEN_EXCLUDE_PATTERNS)
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
  "${PROJECT_BINARY_DIR}/include/corecel/Version.hh"
  "${PROJECT_BINARY_DIR}/include/corecel/Config.hh"
)
if((CELERITAS_DEBUG AND CELERITAS_BUILD_TESTS)
  OR CELERITAS_DOXYGEN_BUILD_TESTS)
  # Add documentation for Celeritas test harnesses when debugging or for CI
  list(APPEND _DOXYGEN_TEST_SOURCE
    "${PROJECT_SOURCE_DIR}/test"
  )
  list(APPEND DOXYGEN_EXCLUDE_PATTERNS
    "*.test.hh" "*.test.cc" "*.test.cu" "*TestBase.cc"
  )
else()
  set(_DOXYGEN_TEST_SOURCE)
endif()

# Include paths must be used for macro preprocessing to function correctly...
set(DOXYGEN_INCLUDE_PATH
  "${PROJECT_BINARY_DIR}/include/"
  "${PROJECT_SOURCE_DIR}/src/"
)

# Documentation usage
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_ENABLED_SECTIONS "CELERITAS_DOC_DEV")
set(DOXYGEN_EXTRA_PACKAGES "amsmath")
set(DOXYGEN_FORMULA_MACROFILE "${CMAKE_CURRENT_SOURCE_DIR}/_static/macros.tex")
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_static")
set(DOXYGEN_INTERNAL_DOCS YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_PREDEFINED "__DOXYGEN__=${_doxy_version_hex}")
set(DOXYGEN_QT_AUTOBRIEF YES)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")

set(DOXYGEN_ALIASES
  "cite{1}=\"[\\1]\""
  "citep{2}=\"([\\1](\\2))\""
  "citet{2}=\"[\\1](\\2)\""
  "rst=\"^^\\if (0)^^\""
  "endrst=\"^^\\endif^^\""
  "rstref{2}=\"\\1 (\\2 in user manual)\""
)

# Verbosity/warning levels
celeritas_set_default(DOXYGEN_QUIET YES)
celeritas_set_default(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

# Output options
set(DOXYGEN_PROJECT_LOGO
  "${CMAKE_CURRENT_SOURCE_DIR}/_static/celeritas-square.svg")
set(DOXYGEN_HTML_EXTRA_STYLESHEET
  "${CMAKE_CURRENT_SOURCE_DIR}/_static/doxygen-custom.css")
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_HTML_OUTPUT "doxygen-html")
set(DOXYGEN_CREATE_SUBDIRS YES)
set(DOXYGEN_REFERENCES_LINK_SOURCE YES)

celeritas_set_default(DOXYGEN_COLLABORATION_GRAPH NO)
celeritas_set_default(DOXYGEN_DOT_GRAPH_MAX_NODES 32)
celeritas_set_default(DOXYGEN_GRAPHICAL_HIERARCHY NO)
celeritas_set_default(DOXYGEN_INCLUDE_GRAPH NO)
celeritas_set_default(DOXYGEN_INCLUDED_BY_GRAPH NO)
celeritas_set_default(DOXYGEN_MAX_DOT_GRAPH_DEPTH 2)
celeritas_set_default(DOXYGEN_VERBATIM_HEADERS NO)

doxygen_add_docs(doxygen
  "${PROJECT_SOURCE_DIR}/src"
  ${_DOXYGEN_EXTRA_SOURCE}
  ${_DOXYGEN_TEST_SOURCE}
  COMMENT "Generating Doxygen to doc/${DOXYGEN_HTML_OUTPUT}/index.html"
)

#-----------------------------------------------------------------------------#
# USER MANUAL
#-----------------------------------------------------------------------------#

if(CELERITAS_PYTHONPATH_RERUN)
  message(SEND_ERROR "The last command must be manually rerun because "
    "Sphinx was not found.")
  unset(CELERITAS_PYTHONPATH_RERUN CACHE)
endif()


if(NOT Sphinx_FOUND)
  function(celeritas_build_sphinx type output depends)
    add_custom_command(
      OUTPUT "${output}"
      COMMAND
        "${CMAKE_COMMAND}"
        -UCELERITAS_PYTHONPATH
        -UCELERITAS_CHECK_PYTHON_MODULE_sphinx
        -DCELERITAS_PYTHONPATH_RERUN:BOOL=ON
        "${CMAKE_BINARY_DIR}"
      COMMENT "Sphinx not found: rerunning CMake with clean PYTHONPATH"
      VERBATIM
    )
  endfunction()
else()
  set(CELERITAS_SPHINX_ARGS "" CACHE STRING
    "Additional arguments to pass to Sphinx to build user docs")
  mark_as_advanced(CELERITAS_SPHINX_ARGS)

  celeritas_get_pyenv(CELER_PYTHONENV)
  # Build sphinx docs:
  # - type: html, pdf, dummy
  # - output: relative path of main output file (index.html, Celeritas.tex)
  # - depends: list of file dependencies
  function(celeritas_build_sphinx type output depends)
    if(NOT type STREQUAL "dummy")
      set(_comment "Building Sphinx ${type} documentation: doc/${output}")
    else()
      set(_comment "Processing Sphinx documentation")
    endif()
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${output}"
      COMMAND
        "${CMAKE_COMMAND}" -E env ${CELER_PYTHONENV}
        "$<TARGET_FILE:Python::Interpreter>" -m sphinx.cmd.build -q
          -d "${CMAKE_CURRENT_BINARY_DIR}/doctrees"
          -b ${type}
          ${CELERITAS_SPHINX_ARGS}
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}/${type}"
      COMMENT "${_comment}"
      BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/${type}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
        "${CMAKE_CURRENT_BINARY_DIR}/config.json"
         ${depends}
      VERBATIM
    )
  endfunction()
endif()

set(_ORNLTM_BASE "${CMAKE_CURRENT_BINARY_DIR}/latex/ornltm")
function(celeritas_build_latex input output)
  set(_texenv
    "LATEXOPTS="
    "TEXINPUTS=${_ORNLTM_BASE}/ornltm:"
  )
  if(NOT LATEXMK_EXECUTABLE)
    set(LATEXMK_EXECUTABLE "latexmk")
  endif()
  add_custom_command(OUTPUT "${output}"
    COMMAND
      "${CMAKE_COMMAND}" -E env ${_texenv}
      "${LATEXMK_EXECUTABLE}" -f -pdf "${input}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/latex"
    VERBATIM
    ${ARGN}
  )
endfunction()

#-----------------------------------------------------------------------------#
# USER MANUAL: Doxygen setup

function(_set_jsonbool varname)
  if(ARGC GREATER 1 AND ARGV1)
    set(${varname} true PARENT_SCOPE)
  else()
    set(${varname} false PARENT_SCOPE)
  endif()
endfunction()

# Define a configure file with version and configuration info
_set_jsonbool(CELERITAS_USE_BREATHE ${CELERITAS_USE_Breathe})
_set_jsonbool(CELERITAS_USE_FURO ${CELERITAS_USE_Furo})
_set_jsonbool(CELERITAS_USE_SPHINXBIBTEX ${CELERITAS_USE_SphinxBibtex})
_set_jsonbool(CELERITAS_USE_DOT ${DOXYGEN_DOT_EXECUTABLE})

if(Celeritas_VERSION STREQUAL Celeritas_VERSION_STRING)
  set(Celeritas_SHORT_VERSION "${Celeritas_VERSION}")
elseif("${Celeritas_VERSION_STRING}" MATCHES
    "^(.*-rc[.-][0-9]+)(\\.[0-9]+\\+[0-9a-f]+)$")
  # Use release candidate, stripping off stuff since the tag
  set(Celeritas_SHORT_VERSION "${CMAKE_MATCH_1}")
else()
  # Development version
  set(_patch "${PROJECT_VERSION_PATCH}")
  if(NOT Celeritas_VERSION_STRING MATCHES "-dev")
    # Before the next release
    math(EXPR _patch "${PROJECT_VERSION_PATCH} + 1")
  endif()
  set(Celeritas_SHORT_VERSION
    "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${_patch}-dev"
  )
endif()
configure_file("config.json.in" "config.json" @ONLY)

set(DOXYGEN_ENABLED_SECTIONS "CELERITAS_DOC_USER")
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_XML_OUTPUT "doxygen-xml")
set(DOXYGEN_XML_PROGRAMLISTING NO)
set(DOXYGEN_COLLABORATION_GRAPH NO)
set(DOXYGEN_CLASS_GRAPH NO)
set(DOXYGEN_INCLUDED_BY_GRAPH NO)
set(DOXYGEN_INCLUDE_GRAPH NO)
set(DOXYGEN_INTERNAL_DOCS NO)

set(DOXYGEN_ALIASES
  "cite{1}=\"\\verbatim embed:rst:inline :cite:p:`\\1` \\endverbatim \""
  "citep{2}=\"\\verbatim embed:rst:inline :cite:p:`\\1` \\endverbatim \""
  "citet{2}=\"\\verbatim embed:rst:inline :cite:t:`\\1` \\endverbatim \""
  "rst=\"^^\\verbatim embed:rst^^\""
  "endrst=\"^^\\endverbatim \""
  "rstref{2}=\"\\verbatim embed:rst:inline :ref:`\\1 <\\2>` \\endverbatim \""
)

# Annotations break breathe parsing
foreach(_var
  CELER_FUNCTION CELER_FORCEINLINE CELER_FORCEINLINE_FUNCTION CELER_CONSTEXPR_FUNCTION
  [[maybe_unused]] CFIF_)
  list(APPEND DOXYGEN_PREDEFINED "${_var}=")
endforeach()

# Only process top two levels (avoid most detail)
file(GLOB _DOXYGEN_SOURCE
  "${PROJECT_SOURCE_DIR}/src/*/*.hh"
  "${PROJECT_SOURCE_DIR}/src/*/*.cc"
  "${PROJECT_SOURCE_DIR}/src/*/*/*.hh"
  "${PROJECT_SOURCE_DIR}/src/corecel/random/data/*.hh"
  "${PROJECT_SOURCE_DIR}/src/corecel/random/engine/*.hh"
  "${PROJECT_SOURCE_DIR}/src/corecel/random/distribution/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/decay/interactor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/data/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/interactor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/interactor/detail/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/model/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/distribution/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/msc/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/msc/detail/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/process/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/em/xs/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/data/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/executor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/interactor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/model/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/model/detail/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/mucf/process/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/neutron/interactor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/neutron/interactor/detail/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/neutron/model/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/optical/gen/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/optical/interactor/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/optical/model/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/optical/surface/*.hh"
  "${PROJECT_SOURCE_DIR}/src/celeritas/optical/surface/model/*.hh"
  "${PROJECT_SOURCE_DIR}/test/*.hh"
)

# IMPORTANT: the target name and USE_STAMP_FILE generate a dependency used by
# the downstream sphinx docs when Breathe is available
doxygen_add_docs(doxygen_xml
  ${_DOXYGEN_SOURCE}
  ${_DOXYGEN_EXTRA_SOURCE}
  USE_STAMP_FILE
  COMMENT "Generating Doxygen: doc/${DOXYGEN_XML_OUTPUT}/index.xml"
)

#-----------------------------------------------------------------------------#
# USER MANUAL: graphics generation (GraphViz)

# Note that *all* dot files must be present at the same time to prevent
# errors when doing separate HTML/PDF builds: the doctrees directory caches the
# file listing and will fail if only one image type is generated
set(_dot_output)
set(_png_fallback "${CMAKE_CURRENT_SOURCE_DIR}/_static/dot-not-found.png")
set(_output_base "${CMAKE_CURRENT_SOURCE_DIR}/_static/dot")

macro(_celer_build_dot inp)
  get_filename_component(basename "${inp}" NAME_WE)
  if(DOXYGEN_DOT_EXECUTABLE)
    # Generate PNG and PDF for HTML, PDF respectively
    foreach(_fmt pdf png)
      set(_output "${_output_base}/${basename}.${_fmt}")
      add_custom_command(
        OUTPUT "${_output}"
        COMMAND
          "${DOXYGEN_DOT_EXECUTABLE}" "-T${_fmt}" "-o${_output}" "${inp}"
        COMMENT "Generating dot graph: ${basename}.${_fmt}"
        DEPENDS "${inp}"
        VERBATIM
      )
      list(APPEND _dot_output "${_output}")
    endforeach()
  else()
    # Fallback: copy placeholder and use for both HTML and PDF
    set(_output "${_output_base}/${basename}.png")
    add_custom_command(
      OUTPUT "${_output}"
      COMMAND
      "${CMAKE_COMMAND}" -E copy "${_png_fallback}" "${_output}"
      COMMENT "Copying fallback graph PNG: ${basename}.png"
      DEPENDS "${inp}"
      VERBATIM
    )
    list(APPEND _dot_output "${_output}")
  endif()
endmacro()

file(GLOB _dot_input "${CMAKE_CURRENT_SOURCE_DIR}/_input/*.dot")
foreach(_inp IN LISTS _dot_input)
  _celer_build_dot("${_inp}")
endforeach()

#-----------------------------------------------------------------------------#
# USER MANUAL: Sphinx RST pre-processing

if(Sphinx_FOUND)
  file(GLOB_RECURSE _sphinx_deps "${CMAKE_CURRENT_SOURCE_DIR}/*.rst")
  list(APPEND _sphinx_deps ${_dot_output})
else()
  set(_sphinx_deps)
endif()

if(CELERITAS_USE_BREATHE)
  list(APPEND _sphinx_deps "${CMAKE_CURRENT_BINARY_DIR}/doxygen_xml.stamp")
endif()

# The built-in 'dummy' builder simply parses the input and caches everything
set(_doctree_env "doctrees/environment.pickle")
celeritas_build_sphinx(dummy "${_doctree_env}" "${_sphinx_deps}")
add_custom_target(doc_preprocess DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_doctree_env}")

#-----------------------------------------------------------------------------#
# USER MANUAL: Sphinx HTML generation

set(_doc_html "html/index.html")
celeritas_build_sphinx(html "${_doc_html}" "${_doctree_env}")
add_custom_target(doc DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_doc_html}")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}"
  COMPONENT documentation
)

#-----------------------------------------------------------------------------#
# USER MANUAL: Sphinx PDF generation

set(_doc_tex "latex/Celeritas.tex")
celeritas_build_sphinx(latex "${_doc_tex}" "${_doctree_env}")

if(Sphinx_FOUND)
  # Download ornltm class
  ExternalProject_Add(ornltm
    GIT_REPOSITORY https://code.ornl.gov/sethrj/ornltm.git
    GIT_TAG 6379032d2f1e5e8bc1589d087465371fd8b432c1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR "${_ORNLTM_BASE}"
  )

  # Generate header
  set(_header_pdf "${CMAKE_CURRENT_BINARY_DIR}/latex/ornltm-header-celeritas.pdf")
  celeritas_build_latex(
    "${CMAKE_CURRENT_SOURCE_DIR}/_static/ornltm-header-celeritas.tex"
    "${_header_pdf}"
    COMMENT "Building tech memo header"
    DEPENDS ornltm
  )

  # Generate footer
  set(_footer_pdf "${CMAKE_CURRENT_BINARY_DIR}/latex/ornltm-footer.pdf")
  celeritas_build_latex(
    "${CMAKE_CURRENT_SOURCE_DIR}/_static/ornltm-footer.tex"
    "${_footer_pdf}"
    COMMENT "Building tech memo footer"
    DEPENDS ornltm
  )

  # Generate PDF
  set(_doc_pdf "latex/Celeritas.pdf")
  celeritas_build_latex(
    Celeritas
    "${CMAKE_CURRENT_BINARY_DIR}/${_doc_pdf}"
    COMMENT "Building final LaTeX document doc/${_doc_pdf}"
    DEPENDS "${_doc_tex}" "${_header_pdf}" "${_footer_pdf}" ornltm
  )
else()
  # No sphinx available: don't confuse people by downloading PDF classes
  set(_doc_pdf "${_doc_tex}")
endif()

add_custom_target(doc_pdf DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_doc_pdf}")

#-----------------------------------------------------------------------------#
