#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasAddTest
----------------

Build and add unit tests for Celeritas.

Commands
^^^^^^^^

.. command:: celeritas_setup_tests

  Set dependencies for the python tests in the current CMakeLists file,
  always resetting the num_process option (see the Variables
  section below) but leaving the link/dependency options in place.

    celeritas_setup_tests(
      [LINK_LIBRARIES <target> [<target> ...]]
      [ADD_DEPENDENCIES <target>]
      [PREFIX <package_name>]
      [SERIAL]
      [PYTHON]
      )

    ``LINK_LIBRARIES``
      Link test executables to these library.

    ``ADD_DEPENDENCIES``
      Require the listed dependencies to be built before the tests.

    ``PREFIX``
      Add the given prefix to the constructed test name.

    ``SERIAL``
      Set CELERITASTEST_NP to 1, so that tests are serial by default.

    ``PYTHON``
      Build Python-basted tests.

.. command:: celeritas_add_test

Add a CUDA/HIP/C++ GoogleTest test::

    celeritas_add_test(
      <filename>
      [TIMEOUT seconds]
      [NP n1 [n2 ...]]
      [LINK_LIBRARIES lib1 [lib2 ...]]
      [DEPTEST deptest]
      [SUFFIX text]
      [ADD_DEPENDENCIES target1 [target2 ...]]
      [ARGS arg1 [arg2 ...]]
      [ENVIRONMENT VAR=value [VAR2=value2 ...]]
      [FILTER test1 [test2 ...]]
      [INPUTS file1 [file2 ...]]
      [SOURCES file1 [...]]
      [DISABLE]
      [REUSE_EXE]
      [GPU]
      )

    ``<filename>``
      Test source file name (or python filename).

    ``NP``
      A list of the number of processes to launch via MPI for this unit test.
      The default is to use CELERITASTEST_NP (1, 2, and 4) for MPI builds and 1
      for serial builds.

    ``NT``
      The number of threads to reserve for this test and set
      ``OMP_NUM_THREADS``, ignored if OpenMP is disabled.

    ``LINK_LIBRARIES``
      Extra libraries to link to. By default, unit tests will link against the
      package's current library.

    ``ADD_DEPENDENCIES``
      Extra dependencies for building the executable, e.g.  preprocessing data
      or copying files.

    ``DEPTEST``
      The base name of another test in the current CMakeLists file that must be
      run before the current test.

    ``SUFFIX``
      Add this suffix to the target and test name.

    ``ENVIRONMENT``
      Set the given environment variables when the test is run.

    ``FILTER``
      A list of ``--gtest_filter`` arguments that will be iterated over.  This
      allows one large test file to be executed as several independent CTest
      tests.

    ``DISABLE``
      Omit this test from the list of tests to run through CTest, but still
      build it to reduce the chance of code rot.

    ``REUSE_EXE``
      Assume the executable was already built from a previous celeritas_add_test
      command. This is useful for specifying combinations of NP/FILTER.

    ``GPU``
      Add a resource lock so that only one GPU test will be run at once.

    ``ADDED_TESTS``
      Output variable name for the name or list of names for added tests,
      suitable for ``set_tests_properties``.

Variables
^^^^^^^^^

The following ``CELERITASTEST_<VAR>`` variables set default properties on tests
defined by ``celeritas_add_test``, analogously to how some ``CMAKE_<VAR>``
variables set default values for target property ``<VAR>``:

``CELERITASTEST_NP`` : list
  Default to running parallel tests with these numbers of processes (default:
  ``1;2;4``).

``CELERITASTEST_LINK_LIBRARIES`` : list
  Link these libraries into each tests.

``CELERITASTEST_ADD_DEPENDENCIES`` : list
  Require that these targets be built before the test is built.

``CELERITASTEST_PYTHONPATH`` : list
  Default entries for the PYTHONPATH environment variable. This should have the
  build directories first, then configure-time directories, then user
  directories.

See the `CMake variable documentation`_ for a description of how scoping will
affect variables. For example, these can be set at global or package level and
overridden in each test directory.

.. _CMake variable documentation : https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cmake-language-variables

#]=======================================================================]

include_guard()

#-----------------------------------------------------------------------------#

set(_procs 1)
if(CELERITAS_USE_MPI)
  list(APPEND _procs 2)
  if(MPIEXEC_MAX_NUMPROCS GREATER 2)
    list(APPEND _procs ${MPIEXEC_MAX_NUMPROCS})
  endif()
endif()
set(CELERITASTEST_NP_DEFAULT "${_procs}" CACHE INTERNAL
  "Default number of processes to use in CeleritasAddTest")
set(_procs)

if(NOT CELERITAS_USE_MPI)
  # Skip tests that require greater than 1 process
  function(_celeritasaddtest_test_name outvar test_name np suffix)
    set(_name "${test_name}${suffix}")
    if(np GREATER 1)
      set(_name)
    endif()
    set(${outvar} "${_name}" PARENT_SCOPE)
  endfunction()

  # Construct execution command
  function(_celeritasaddtest_mpi_cmd outvar np test_exe)
    set(_cmd "${test_exe}" ${ARGN})
    set(${outvar} "${_cmd}" PARENT_SCOPE)
  endfunction()
else()
  # Construct test name with number of processors
  function(_celeritasaddtest_test_name outvar test_name np suffix)
    if(np GREATER CELERITAS_MAX_NUMPROCS)
      set(_name)
    elseif(np GREATER 1)
      set(_name "${test_name}/${np}${suffix}")
    else()
      set(_name "${test_name}${suffix}")
    endif()
    set(${outvar} "${_name}" PARENT_SCOPE)
  endfunction()

  # Construct MPI command
  function(_celeritasaddtest_mpi_cmd outvar np test_exe)
    if(np GREATER 1)
      set(_cmd "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} "${np}"
        ${MPIEXEC_PREFLAGS} ${test_exe} ${MPIEXEC_POSTFLAGS} ${ARGN})
    else()
      set(_cmd ${test_exe} ${ARGN})
    endif()
    set(${outvar} "${_cmd}" PARENT_SCOPE)
  endfunction()
endif()

#-----------------------------------------------------------------------------#
# celeritas_setup_tests
#-----------------------------------------------------------------------------#

function(celeritas_setup_tests)
  cmake_parse_arguments(PARSE
    "SERIAL;PYTHON"
    "PREFIX"
    "LINK_LIBRARIES;ADD_DEPENDENCIES"
    ${ARGN}
  )

  # Set the directory relative to the "test" directory
  file(RELATIVE_PATH PARSE_DIR
    "${PROJECT_SOURCE_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}")

  # Set special variables
  foreach(_var LINK_LIBRARIES ADD_DEPENDENCIES PREFIX DIR)
    set(CELERITASTEST_${_var} "${PARSE_${_var}}" PARENT_SCOPE)
  endforeach()

  # Override default num procs if requested
  set(CELERITASTEST_NP ${CELERITASTEST_NP_DEFAULT})
  if(PARSE_SERIAL)
    set(CELERITASTEST_NP 1)
  endif()
  set(CELERITASTEST_NP "${CELERITASTEST_NP}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
# celeritas_add_test
#-----------------------------------------------------------------------------#

function(celeritas_add_test SOURCE_FILE)
  cmake_parse_arguments(PARSE
    "DISABLE;REUSE_EXE;GPU"
    "TIMEOUT;DEPTEST;SUFFIX;ADDED_TESTS;NT"
    "LINK_LIBRARIES;ADD_DEPENDENCIES;NP;ENVIRONMENT;ARGS;INPUTS;FILTER;SOURCES"
    ${ARGN}
  )
  if(PARSE_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unknown keywords given to celeritas_add_test(): "
            "\"${PARSE_UNPARSED_ARGUMENTS}\"")
  endif()

  if(PARSE_INPUTS)
    message(FATAL_ERROR "INPUTS argument to celeritas_add_test is not implemented")
  endif()

  if(NOT PARSE_NP)
    if(CELERITASTEST_NP)
      set(PARSE_NP ${CELERITASTEST_NP})
    else()
      set(PARSE_NP ${CELERITASTEST_NP_DEFAULT})
    endif()
  endif()

  # Add prefix to test name and possibly dependent name
  get_filename_component(_BASENAME "${SOURCE_FILE}" NAME_WE)
  get_filename_component(_BASEDIR "${SOURCE_FILE}" DIRECTORY)
  if(_BASEDIR)
    set(_BASENAME "${_BASEDIR}/${_BASENAME}")
  endif()
  set(_TARGET "${_BASENAME}")
  if(CELERITASTEST_PREFIX)
    set(_TARGET "${CELERITASTEST_PREFIX}/${_TARGET}")
    if(PARSE_DEPTEST)
      set(PARSE_DEPTEST "${CELERITASTEST_PREFIX}/${PARSE_DEPTEST}")
    endif()
    set(_PREFIX "${CELERITASTEST_PREFIX}/")
  endif()
  set(_OUTPUT_NAME "${_TARGET}")
  if(CELERITASTEST_DIR)
    set(_TARGET "${CELERITASTEST_DIR}/${_TARGET}")
    if(PARSE_DEPTEST)
      set(PARSE_DEPTEST "${CELERITASTEST_DIR}/${PARSE_DEPTEST}")
    endif()
    set(_PREFIX "${CELERITASTEST_DIR}/${CELERITASTEST_PREFIX}")
  endif()

  if(PARSE_SUFFIX)
    set(_TARGET "${_TARGET}_${PARSE_SUFFIX}")
    if(PARSE_DEPTEST)
      set(PARSE_DEPTEST "${PARSE_DEPTEST}/${PARSE_SUFFIX}")
    endif()
    set(_SUFFIX "/${PARSE_SUFFIX}")
  endif()

  string(REGEX REPLACE "[^a-zA-Z0-9_]+" "_" _TARGET "${_TARGET}")
  string(REGEX REPLACE "[^a-zA-Z0-9_]+" "_" _OUTPUT_NAME "${_OUTPUT_NAME}")

  if(PARSE_REUSE_EXE)
    set(_EXE_NAME "$<TARGET_FILE:${_TARGET}>")
    if(NOT TARGET "${_TARGET}")
      message(WARNING "Target ${_TARGET} has not yet been created")
    endif()
  else()
    set(_EXE_NAME "$<TARGET_FILE:${_TARGET}>")

    # Create an executable and link libraries against it
    add_executable(${_TARGET} "${SOURCE_FILE}" ${PARSE_SOURCES})

    set_property(TARGET ${_TARGET}
      PROPERTY OUTPUT_NAME "${_OUTPUT_NAME}"
    )

    # Note: for static linking the library order is relevant.
    celeritas_target_link_libraries(${_TARGET}
      ${CELERITASTEST_LINK_LIBRARIES}
      ${PARSE_LINK_LIBRARIES}
      testcel_harness
    )

    if(PARSE_ADD_DEPENDENCIES OR CELERITASTEST_ADD_DEPENDENCIES)
      # Add additional dependencies
      add_dependencies(${_TARGET} ${PARSE_ADD_DEPENDENCIES}
        ${CELERITASTEST_ADD_DEPENDENCIES})
    endif()
  endif()

  # Add standard CELERITAS environment variables
  if(CELERITASTEST_PYTHONPATH)
    list(APPEND PARSE_ENVIRONMENT "PYTHONPATH=${CELERITASTEST_PYTHONPATH}")
  endif()

  if(CELERITAS_TEST_VERBOSE)
    list(APPEND PARSE_ENVIRONMENT
      "CELER_LOG=debug"
      "CELER_LOG_LOCAL=diagnostic"
    )
  else()
    list(APPEND PARSE_ENVIRONMENT
      "CELER_LOG=warning"
      "CELER_LOG_LOCAL=warning"
    )
  endif()

  set(_COMMON_PROPS
    PASS_REGULAR_EXPRESSION "tests PASSED"
    FAIL_REGULAR_EXPRESSION "tests FAILED"
  )
  set(_LABELS)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    if(NOT PARSE_GPU)
      list(APPEND PARSE_ENVIRONMENT "CELER_DISABLE_DEVICE=1")
    else()
      if(CELERITAS_TEST_RESOURCE_LOCK)
        # Add a "resource lock" when building without debug checking to get more
        # accurate test times (since multiple GPU processes won't be competing for
        # the same card).
        # To speed up overall test time, *do not* apply the resource lock when
        # we're debugging because timings don't matter.
        list(APPEND _COMMON_PROPS RESOURCE_LOCK gpu)
      endif()
      list(APPEND _LABELS gpu)
    endif()
  endif()
  if(PARSE_SOURCES AND CELERITAS_USE_HIP)
    celeritas_get_cuda_source_args(_cuda_sources ${PARSE_SOURCES})
    if(_cuda_sources)
      # When building Celeritas libraries, we put HIP/CUDA files in shared .cu
      # suffixed files. Override the language if using HIP.
      set_source_files_properties(
        ${_cuda_sources}
        PROPERTIES LANGUAGE HIP
      )
    endif()
  endif()
  if(PARSE_TIMEOUT)
    list(APPEND _COMMON_PROPS TIMEOUT ${PARSE_TIMEOUT})
  endif()
  if(PARSE_DISABLE)
    list(APPEND _COMMON_PROPS DISABLED TRUE)
  endif()
  list(APPEND _LABELS unit)

  if(CELERITAS_USE_MPI AND PARSE_NP STREQUAL "1")
    list(APPEND PARSE_ENVIRONMENT "CELER_DISABLE_PARALLEL=1")
  endif()

  if(CELERITAS_USE_OpenMP)
    if(PARSE_NT)
      list(APPEND PARSE_ENVIRONMENT "OMP_NUM_THREADS=${PARSE_NT}")
    endif()
  else()
    set(PARSE_NT)
  endif()

  if(NOT PARSE_FILTER)
    # Set to a non-empty but false value
    set(PARSE_FILTER "OFF")
  endif()

  foreach(_filter IN LISTS PARSE_FILTER)
    foreach(_np IN LISTS PARSE_NP)
      set(_suffix)
      if(_filter)
        set(_suffix ":${_filter}")
      endif()
      _celeritasaddtest_test_name(_TEST_NAME
        "${_PREFIX}${_BASENAME}${_SUFFIX}" "${_np}" "${_suffix}")
      if(NOT _TEST_NAME)
        continue()
      endif()

      set(_test_env ${PARSE_ENVIRONMENT})

      # Launch with MPI directly
      _celeritasaddtest_mpi_cmd(_test_cmd "${_np}" "${_EXE_NAME}")

      set(_test_args)
      if(_filter)
        list(APPEND _test_args "--gtest_filter=${_filter}")
      endif()
      if(CELERITAS_TEST_XML)
        string(REGEX REPLACE "\\*" "ALL" _xml_name "${_TEST_NAME}")
        string(REGEX REPLACE "[^a-zA-Z0-9_.-]+" "_" _xml_name "${_xml_name}")
        list(APPEND _test_args
          "--gtest_output=xml:${CELERITAS_TEST_XML}/${_xml_name}.xml"
        )
      endif()

      add_test(NAME "${_TEST_NAME}" COMMAND ${_test_cmd} ${_test_args})
      list(APPEND _ADDED_TESTS "${_TEST_NAME}")

      if(PARSE_DEPTEST)
        # Add dependency on another test
        _celeritasaddtest_test_name(_deptest_name
          "${PARSE_DEPTEST}" "${_np}" "${_suffix}")
        set_property(TEST "${_TEST_NAME}"
          PROPERTY ADD_DEPENDENCIES "${_deptest_name}")
      endif()

      set_property(TEST ${_TEST_NAME}
        PROPERTY ENVIRONMENT ${_test_env}
      )
      if(_np GREATER 1)
        if(PARSE_NT)
          math(EXPR _np "${_np} * ${PARSE_NT}")
        endif()
        set_property(TEST ${_TEST_NAME}
          PROPERTY PROCESSORS ${_np}
        )
      endif()
    endforeach()
  endforeach()

  if(_ADDED_TESTS)
    # Set common properties
    set_tests_properties(${_ADDED_TESTS}
      PROPERTIES ${_COMMON_PROPS}
      "LABELS" "${_LABELS}")
  endif()
  if(PARSE_ADDED_TESTS)
    # Export test names
    set(${PARSE_ADDED_TESTS} ${_ADDED_TESTS} PARENT_SCOPE)
  endif()
endfunction()

#-----------------------------------------------------------------------------#
