#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasG4Tests
----------------

Add all available permutations of tests for executables managed by the Geant4
run manager.

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
  )

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

function(celeritas_g4_add_one_test test_name target args labels offload rmtype)
  add_test(NAME "${test_name}" COMMAND "$<TARGET_FILE:${target}>" ${args})
  set(_env
    ${_celer_g4_test_env}
    "G4RUN_MANAGER_TYPE=${_rm_${rmtype}}"
  )
  if(offload STREQUAL "cpu" AND (CELERITAS_USE_CUDA OR CELERITAS_USE_HIP))
    list(APPEND _env "CELER_DISABLE_DEVICE=1")
  elseif(offload STREQUAL "gpu")
    list(APPEND _extra_props RESOURCE_LOCK gpu)
    list(APPEND labels gpu)
  elseif(offload STREQUAL "g4")
    list(APPEND _env "CELER_DISABLE=1")
  elseif(offload STREQUAL "ko")
    list(APPEND _env "CELER_KILL_OFFLOAD=1")
  endif()
  if(NOT rmtype STREQUAL "serial")
    list(APPEND _env "G4FORCENUMBEROFTHREADS=2")
    list(APPEND _extra_props PROCESSORS 2)
  endif()

  set_tests_properties("${test_name}" PROPERTIES
    ENVIRONMENT "${_env}"
    LABELS "${labels}"
    ${_extra_props}
    ${ARGN}
  )
endfunction()

function(celeritas_g4_add_tests target)
  cmake_parse_arguments(PARSE "DISABLE" "NAME" "ARGS;LABELS" ${ARGN})
  if(PARSE_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unknown keywords given to celeritas_g4_add_tests(): "
            "\"${PARSE_UNPARSED_ARGUMENTS}\"")
  endif()

  if(NOT PARSE_NAME)
    set(PARSE_NAME ${target})
  endif()
  foreach(_offload gpu cpu g4 ko)
    foreach(_rmtype serial mt task)
      set(_disable ${PARSE_DISABLE})
      if(_offload STREQUAL "gpu" AND _disable_device)
        set(_disable TRUE)
      endif()
      if(NOT _rm_avail_${_rmtype})
        set(_disable TRUE)
      endif()
      if(_disable)
        set(_args DISABLED TRUE)
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
