#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasG4Tests
----------------

Add all available permutations of tests for executables managed by the Geant4
run manager.

.. warning:: This cmake file *must* be included in the same scope as it is used
   due to internal directory-scoped variables. This requirement may be relaxed
   in the future.

.. command:: celeritas_add_g4_tests

  This will add all permutations of available G4 run managers (serial, MT,
  tasking) and Celeritas runtimes (cpu, gpu, disable)::

  celeritas_g4_add_tests(
    <target>
    [PREFIX string]      # before target in test name
    [SUFFIX string]      # after target in test name
    [ARGS arg [...]]     # passed to command line
    [LABELS label [...]] # tagged in CTestFile
    [DISABLE]            # display but don't run
    [WILL_FAIL]          # expect the test to exit with a failure
  )

  Note that DISABLE silently overrides WILL_FAIL.

#]=======================================================================]

include_guard()

include(CeleritasUtils)

if(NOT DEFINED Geant4_VERSION OR NOT DEFINED Celeritas_VERSION)
  message(FATAL_ERROR
    "This file must be included after Celeritas and Geant4 have been found"
  )
endif()

celeritas_get_g4env(_celer_g4_test_env)
celeritas_get_disable_device(_disable_device)

if(CELERITAS_USE_MPI)
  list(APPEND _celer_g4_test_env "CELER_DISABLE_PARALLEL=1")
endif()

# Get correct capitalization for run manager type
set(_rm_serial Serial)
set(_rm_mt MT)
set(_rm_task Tasking)

# Set whether run manager is available
set(_rm_avail_serial TRUE)
set(_rm_avail_mt TRUE)
set(_rm_avail_task TRUE)
if(NOT Geant4_multithreaded_FOUND)
  set(_rm_avail_mt FALSE)
endif()
if(Geant4_VERSION VERSION_LESS 11.0)
  set(_rm_avail_task FALSE)
endif()

# Set number of threads allowed
set(_rm_threads 2)

function(celeritas_g4_add_one_test test_name target args labels offload rmtype)
  add_test(NAME "${test_name}" COMMAND "$<TARGET_FILE:${target}>" ${args})
  if(NOT DEFINED _rm_${rmtype})
    message(SEND_ERROR "Invalid run manager type ${rmtype}")
  endif()
  set(_env
    ${_celer_g4_test_env}
    "CELER_OFFLOAD=${offload}" # Used only by test framework for now
    "G4RUN_MANAGER_TYPE=${_rm_${rmtype}}"
  )
  if(offload STREQUAL "cpu")
    if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
     list(APPEND _env "CELER_DISABLE_DEVICE=1")
    endif()
  elseif(offload STREQUAL "gpu")
    list(APPEND _extra_props RESOURCE_LOCK gpu)
    list(APPEND labels gpu)
  elseif(offload STREQUAL "g4")
    list(APPEND _env "CELER_DISABLE=1")
  elseif(offload STREQUAL "ko")
    list(APPEND _env "CELER_KILL_OFFLOAD=1")
  else()
    message(SEND_ERROR "Invalid offload type ${offload}")
  endif()
  if(NOT rmtype STREQUAL "serial")
    list(APPEND _env "G4FORCENUMBEROFTHREADS=${_rm_threads}")
    list(APPEND _extra_props PROCESSORS ${_rm_threads})
  endif()

  set_tests_properties("${test_name}" PROPERTIES
    ENVIRONMENT "${_env}"
    LABELS "${labels}"
    ${_extra_props}
    ${ARGN}
  )
endfunction()

function(celeritas_g4_add_tests target)
  cmake_parse_arguments(PARSE "DISABLE;WILL_FAIL" "NAME" "ARGS;LABELS;RMTYPE;OFFLOAD" ${ARGN})
  if(PARSE_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unknown keywords given to celeritas_g4_add_tests(): "
            "\"${PARSE_UNPARSED_ARGUMENTS}\"")
  endif()

  if(NOT PARSE_NAME)
    set(PARSE_NAME ${target})
  endif()
  if(NOT PARSE_RMTYPE)
    set(PARSE_RMTYPE serial mt task)
  endif()
  if(NOT PARSE_OFFLOAD)
    set(PARSE_OFFLOAD gpu cpu g4 ko)
  endif()
  foreach(_offload IN LISTS PARSE_OFFLOAD)
    foreach(_rmtype IN LISTS PARSE_RMTYPE)
      set(_disable ${PARSE_DISABLE})
      if(_offload STREQUAL "gpu" AND _disable_device)
        set(_disable TRUE)
      endif()
      if(NOT _rm_avail_${_rmtype})
        set(_disable TRUE)
      endif()
      if(_disable)
        set(_args DISABLED TRUE)
      elseif(PARSE_WILL_FAIL)
        set(_args WILL_FAIL TRUE)
      else()
        set(_args)
      endif()

      set(_test_name "${PARSE_NAME}:${_offload}:${_rmtype}")
      celeritas_g4_add_one_test(
        ${_test_name} ${target} "${PARSE_ARGS}" "${PARSE_LABELS}"
        ${_offload} ${_rmtype} ${_args}
      )
    endforeach()
  endforeach()
endfunction()

#-----------------------------------------------------------------------------#
