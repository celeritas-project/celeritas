##---------------------------------------------------------------------------##
## File  : cmake/CeleritasLoadSubmodule.cmake
#[=======================================================================[.rst:

CeleritasLoadSubmodule
----------------------

Utility commands for the submodule checkouts.

.. command:: celeritas_load_submodule

  Load a Git submodlue at the specified directory::

    celeritas_load_submodule(<subdir>)

  ``subdir``
    Subdirectory of the current directory that is a git subrepositoy.

The behavior of this command depends on the ``CELERITAS_GIT_SUBMODULE``
variable: if false, no git commands will be run. This variable defaults to True
only when the celeritas source directory contains a top-level ``.git`` directory.

When that variable is set, another variable,
``CELERITAS_GIT_SUBMODULE_AGGRESSIVE``, will be exposed: it will always
attempt to update the git submodule checkouts when CMake is run. If unset you
will manually have to update the submodules if the upstream version changes.

#]=======================================================================]


if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
  set(IS_GIT_REPOSITORY TRUE)
  set(_ERROR_MSG "Manually run `git submodule update --init --recursive` to load external dependencies and disable the CELERITAS_GIT_SUBMODULE option.")
else()
  set(IS_GIT_REPOSITORY FALSE)
endif()

##---------------------------------------------------------------------------##
## OPTIONS
##---------------------------------------------------------------------------##

# Clone submodules by default only if we're a git repository
option(CELERITAS_GIT_SUBMODULE
  "Automatically download Git submodules during configuration"
  ${IS_GIT_REPOSITORY})

if(CELERITAS_GIT_SUBMODULE)
  # Default to keeping submodules in sync
  option(CELERITAS_GIT_SUBMODULE_AGGRESSIVE
    "Try to update Git submodules during *every* configuration"
    ON)
endif()

##---------------------------------------------------------------------------##
## SETUP
##---------------------------------------------------------------------------##

if(CELERITAS_GIT_SUBMODULE)
  # Find git
  find_package(Git)
  if(NOT GIT_FOUND OR GIT_VERSION_STRING VERSION_LESS "1.5.3")
    # Git 1.5.3 is the earliest to support submodules.
    message(FATAL_ERROR "Git version 1.5.3 or higher must be available to use "
      "git submodules. ${_ERROR_MSG}")
  endif()
endif()

##---------------------------------------------------------------------------##

function(celeritas_load_submodule SUBDIR)
  if(NOT IS_ABSOLUTE "${SUBDIR}")
    set(SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}")
  endif()
  file(TO_CMAKE_PATH "${SUBDIR}" SUBDIR)

  if(EXISTS "${SUBDIR}/.git")
    set(_IS_GIT_SUBDIR TRUE)
  else()
    set(_IS_GIT_SUBDIR FALSE)
  endif()

  if(_IS_GIT_SUBDIR AND NOT CELERITAS_GIT_SUBMODULE_AGGRESSIVE)
    message(STATUS
      "Git submodule \"${SUBDIR}\" is already checked out")
    return()
  endif()

  # Only clone through cmake if requested
  if(NOT CELERITAS_GIT_SUBMODULE)
    if(IS_GIT_REPOSITORY AND NOT _IS_GIT_SUBDIR)
      message(WARNING "Git submodule checkout is disabled but ${SUBDIR} "
      "appears not to be available.")
    endif()
    return()
  endif()

  # Older git versions must run `git submodule` from the root git work
  # directory.
  file(RELATIVE_PATH SUBMODULE_DIR "${PROJECT_SOURCE_DIR}" "${SUBDIR}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
      "${SUBMODULE_DIR}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ERROR_VARIABLE GIT_SUBMOD_ERR
    OUTPUT_VARIABLE GIT_SUBMOD_MSG
    RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(GIT_SUBMOD_ERR)
    set(GIT_SUBMOD_ERR ":\n  ${GIT_SUBMOD_ERR}")
  endif()
  if(GIT_SUBMOD_MSG)
    set(GIT_SUBMOD_MSG ":\n  ${GIT_SUBMOD_MSG}")
  endif()
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "${GIT_EXECUTABLE} submodule update --init "
      " ${SUBDIR} failed (cwd ${CMAKE_CURRENT_SOURCE_DIR}): "
      "error code ${GIT_SUBMOD_RESULT}${GIT_SUBMOD_ERR}); "
      "${_ERROR_MSG}${GIT_SUBMOD_MSG}")
  else()
    message(STATUS
      "Successfully updated git submodule \"${SUBDIR}\"${GIT_SUBMOD_MSG}")
  endif()

  if(NOT EXISTS "${SUBDIR}/.git")
    message(FATAL_ERROR "git submodule update --init failed to check out "
      "${SUBDIR}: ${_ERROR_MSG}")
  endif()
endfunction()


##---------------------------------------------------------------------------##
## end of cmake/CeleritasLoadSubmodule.cmake
##---------------------------------------------------------------------------##
