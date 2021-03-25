#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasAddTest
----------------

Build and add unit tests for Celeritas.

Commands
^^^^^^^^

.. command:: celeritas_setup_tests

  Set dependencies for the python tests in the current CMakeLists file,
  always resetting the num_process and harness options (see the Variables
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
      Require the listed depenencies to be built before the tests.

    ``PREFIX``
      Add the given prefix to the constructed test name.

    ``SERIAL``
      Set CELERITASTEST_NP to 1, so that tests are serial by default.

    ``PYTHON``
      Build Python-basted tests.

.. command:: celeritas_add_test

Add a CUDA/C++ GoogleTest test::

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
      [ISOLATE]
      [DISABLE]
      [DRIVER]
      [REUSE_EXE]
      [GPU]
      )

    ``<filename>``
      Test source file name (or python filename).

    ``NP``
      The number of processors to use for this unit test. The default
      is to use CELERITASTEST_NP (1, 2, and 4) for MPI builds and 1 for
      serial builds.

    ``LINK_LIBRARIES``
      Extra libraries to link to. By default, unit tests will link against the
      package's current library.

    ``ADD_DEPENDENCIES``
      Extra dependencies for building the execuatable, e.g.  preprocessing data
      or copying files.

    ``DEPTEST``
      The base name of another test in the current CMakeLists file that must be
      run before the current test.

    ``SUFFIX``
      Add this suffix to the target and test name.

    ``ENVRIONMENT``
      Set the given environment variables when the test is run.

    ``FILTER``
      A list of ``--gtest_filter`` arguments that will be iterated over.  This
      allows one large test file to be executed as several independent CTest
      tests.

    ``INPUTS``
      Copy the listed files to the test working directory. (Works best with
      ISOLATE.) TODO: not implemented.

    ``ISOLATE``
      Run the test in its own directory.

    ``DISABLE``
      Omit this test from the list of tests to run through CTest, but still
      build it to reduce the chance of code rot.

    ``DRIVER``
      Assume the file acts as a "driver" that launches an underlying
      process in parallel. The CELERITASTEST_NP environment variable is set for that
      test and can be used to determine the number of processors to use. The
      test itself is *not* called with MPI.

    ``REUSE_EXE``
      Assume the executable was already built from a previous celeritas_add_test
      command. This is useful for specifying combinations of NP/FILTER.

    ``GPU``
      Add a resource lock so that only one GPU test will be run at once.

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

``CELERITASTEST_HARNESS`` : string
  One of "none", "gtest", or "python". Defaults to "gtest" (use the Nemesis gtest
  harness and main function); the "python" option uses the exnihilotools unit
  test harness.

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

if(CELERITAS_USE_MPI)
  set(CELERITASTEST_NP_DEFAULT "1;2;4" CACHE INTERNAL
    "Default number of processes to use in CeleritasAddTest")
else()
  set(CELERITASTEST_NP_DEFAULT "1" CACHE INTERNAL
    "Default number of processes to use in CeleritasAddTest")
endif()

if(NOT CELERITAS_USE_MPI)
  # Construct test name with MPI enabled, or empty if not applicable
  function(_celeritasaddtest_test_name outvar test_name np suffix)
    set(_name "${test_name}${suffix}")
    if(np GREATER 1)
      set(_name)
    endif()
    set(${outvar} "${_name}" PARENT_SCOPE)
  endfunction()

  # Construct MPI command, or empty if not applicable
  function(_celeritasaddtest_mpi_cmd outvar np test_exe)
    set(_cmd "${test_exe}" ${ARGN})
    set(${outvar} "${_cmd}" PARENT_SCOPE)
  endfunction()
else()
  # Construct test name with MPI enabled but not tribits
  function(_celeritasaddtest_test_name outvar test_name np suffix)
    if(np GREATER CELERITAS_MAX_NUMPROCS)
      set(_name)
    elseif(np GREATER 1)
      set(_name "${test_name}/${np}${suffix}")
    else()
      set(_name "${test_name}")
    endif()
    set(${outvar} "${_name}" PARENT_SCOPE)
  endfunction()

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

  # Set special variables
  foreach(_var LINK_LIBRARIES ADD_DEPENDENCIES PREFIX)
    set(CELERITASTEST_${_var} "${PARSE_${_var}}" PARENT_SCOPE)
  endforeach()

  # Override default num procs if requested
  set(CELERITASTEST_NP ${CELERITASTEST_NP_DEFAULT})
  if(PARSE_SERIAL)
    set(CELERITASTEST_NP 1)
  endif()
  set(CELERITASTEST_NP "${CELERITASTEST_NP}" PARENT_SCOPE)

  set(CELERITASTEST_HARNESS "gtest")
  if(PARSE_PYTHON)
    set(CELERITASTEST_HARNESS "python")
  endif()
  set(CELERITASTEST_HARNESS "${CELERITASTEST_HARNESS}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
# celeritas_add_test
#-----------------------------------------------------------------------------#

function(celeritas_add_test SOURCE_FILE)
  cmake_parse_arguments(PARSE
    "ISOLATE;DISABLE;DRIVER;REUSE_EXE;GPU"
    "TIMEOUT;DEPTEST;SUFFIX"
    "LINK_LIBRARIES;ADD_DEPENDENCIES;NP;ENVIRONMENT;ARGS;INPUTS;FILTER;SOURCES"
    ${ARGN}
  )

  if(NOT CELERITASTEST_HARNESS OR CELERITASTEST_HARNESS STREQUAL "gtest")
    set(_CELERITASTEST_IS_GTEST TRUE)
  elseif(CELERITASTEST_HARNESS STREQUAL "python")
    set(_CELERITASTEST_IS_PYTHON TRUE)
  elseif(SOURCE_FILE MATCHES "\.py$")
    set(_CELERITASTEST_IS_PYTHON TRUE)
  endif()

  if(PARSE_INPUTS)
    message(FATAL_ERROR "INPUTS argument to celeritas_add_test is not implemented" )
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
  if(CELERITASTEST_PREFIX)
    set(_TARGET "${CELERITASTEST_PREFIX}/${_BASENAME}")
    string(REGEX REPLACE "[^a-zA-Z0-9_]+" "_" _TARGET "${_TARGET}")
    if(PARSE_DEPTEST)
      set(PARSE_DEPTEST "${CELERITASTEST_PREFIX}/${PARSE_DEPTEST}")
    endif()
    set(_PREFIX "${CELERITASTEST_PREFIX}/")
  endif()
  if(PARSE_SUFFIX)
    set(_TARGET "${_TARGET}_${PARSE_SUFFIX}")
    if(PARSE_DEPTEST)
      set(PARSE_DEPTEST "${PARSE_DEPTEST}/${PARSE_SUFFIX}")
    endif()
    set(_SUFFIX "/${PARSE_SUFFIX}")
  endif()

  if(_CELERITASTEST_IS_PYTHON)
    get_filename_component(SOURCE_FILE "${SOURCE_FILE}" ABSOLUTE)
    set(_EXE_NAME "${PYTHON_EXECUTABLE}")
    set(_EXE_ARGS -W once "${SOURCE_FILE}" -v)
    if(NOT EXISTS "${SOURCE_FILE}")
      message(SEND_ERROR "Python test file '${SOURCE_FILE}' does not exist")
    endif()
    if(PARSE_SOURCES)
      message(FATAL_ERROR "The SOURCE argument cannot be used "
        "with Python tests")
    endif()
  elseif(PARSE_REUSE_EXE)
    set(_EXE_NAME "$<TARGET_FILE:${_TARGET}>")
    if(NOT TARGET "${_TARGET}")
      message(WARNING "Target ${_TARGET} has not yet been created")
    endif()
  else()
    set(_EXE_NAME "$<TARGET_FILE:${_TARGET}>")

    # Create an executable and link libraries against it
    add_executable(${_TARGET} "${SOURCE_FILE}" ${PARSE_SOURCES})
    target_link_libraries(${_TARGET}
      Celeritas::Test Celeritas::Core
      ${CELERITASTEST_LINK_LIBRARIES}
      ${PARSE_LINK_LIBRARIES}
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

  if(CELERITAS_DEBUG)
    list(APPEND PARSE_ENVIRONMENT
      "CELER_LOG=debug"
      "CELER_LOG_LOCAL=diagnostic"
    )
  endif()

  set(_COMMON_PROPS)
  if(CELERITAS_USE_CUDA)
    if(NOT PARSE_GPU)
      list(APPEND PARSE_ENVIRONMENT "CELER_DISABLE_DEVICE=1")
    elseif(NOT CELERITAS_DEBUG)
      # Add a "resource lock" when building without debug checking to get more
      # accurate test times (since multiple GPU processes won't be competing for
      # the same card).
      # To speed up overall test time, *do not* apply the resource lock when
      # we're debugging because timings don't matter.
      list(APPEND _COMMON_PROPS RESOURCE_LOCK gpu)
    endif()
  endif()
  if(PARSE_TIMEOUT)
    list(APPEND _COMMON_PROPS TIMEOUT ${PARSE_TIMEOUT})
  endif()
  if(PARSE_DISABLE)
    list(APPEND _COMMON_PROPS DISABLED True)
  endif()
  if(_CELERITASTEST_IS_GTEST OR _CELERITASTEST_IS_PYTHON)
    list(APPEND _COMMON_PROPS
      PASS_REGULAR_EXPRESSION "tests PASSED"
      FAIL_REGULAR_EXPRESSION "tests FAILED"
    )
  endif()

  if(CELERITAS_USE_MPI AND PARSE_NP STREQUAL "1")
    list(APPEND PARSE_ENVIRONMENT "CELER_DISABLE_PARALLEL=1")
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
      if(NOT PARSE_DRIVER)
        # Launch with MPI directly
        _celeritasaddtest_mpi_cmd(_test_cmd "${_np}" "${_EXE_NAME}")
      else()
        # Just allow the test to determine the number of procs
        set(_test_cmd "${_EXE_NAME}")
        list(APPEND _test_env "CELERITASTEST_NUMPROCS=${_np}")
      endif()

      set(_test_args "${_EXE_ARGS}")
      if(_filter)
        if(_CELERITASTEST_IS_GTEST)
          list(APPEND _test_args "--gtest_filter=${_filter}")
        elseif(_CELERITASTEST_IS_PYTHON)
          list(APPEND _test_args "${_filter}")
        endif()
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

      if(PARSE_ISOLATE)
        # Run in a separate working directory
        string(REGEX REPLACE "[^a-zA-Z0-9-]" "-" _test_dir "${_TEST_NAME}")
        string(TOLOWER "${_test_dir}" _test_dir)
        set(_test_dir "${CMAKE_CURRENT_BINARY_DIR}/run-${_test_dir}")
        set_property(TEST "${_TEST_NAME}"
          PROPERTY WORKING_DIRECTORY "${_test_dir}")
        file(MAKE_DIRECTORY "${_test_dir}")
      endif()

      set_property(TEST ${_TEST_NAME}
        PROPERTY ENVIRONMENT ${_test_env}
      )
      if(_np GREATER 1)
        set_property(TEST ${_TEST_NAME}
          PROPERTY PROCESSORS ${_np}
        )
      endif()
    endforeach()
  endforeach()
  if(_COMMON_PROPS AND _ADDED_TESTS)
    # Set common properties
    set_tests_properties(${_ADDED_TESTS} PROPERTIES ${_COMMON_PROPS})
  endif()
endfunction()

#-----------------------------------------------------------------------------#
