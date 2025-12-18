#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT DD4hep_FOUND)
  message(STATUS "DD4hep not found: skipping ddceler example")
  return()
endif()

if(NOT ROOT_FOUND)
  message(STATUS "ROOT not found: skipping ddceler validation")
  return()
endif()

#-----------------------------------------------------------------------------#
# DD4hep Comparison Executable
#-----------------------------------------------------------------------------#

add_executable(ddceler-compare compare_benchmarks.cc)

target_link_libraries(ddceler-compare
  PRIVATE
    ROOT::Core
    ROOT::Hist
    ROOT::Graf
    ROOT::Gpad
    ROOT::Tree
    ROOT::TreePlayer
    ROOT::RIO
    DD4hep::DDCore
    DD4hep::DDG4
)

target_include_directories(ddceler-compare
  PRIVATE
    ${DD4hep_INCLUDE_DIRS}
    ${ROOT_INCLUDE_DIRS}
)

# Inherit C++ standard from project settings
set_target_properties(ddceler-compare PROPERTIES
  CXX_STANDARD ${CMAKE_CXX_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

install(TARGETS ddceler-compare
  RUNTIME DESTINATION bin
)

message(STATUS "Added DD4hep comparison executable: ddceler-compare")

#-----------------------------------------------------------------------------#
# CTest Integration using Environment Variables
#-----------------------------------------------------------------------------#

# Add validation test that expects environment variables
add_test(
  NAME ddceler-validation
  COMMAND $<TARGET_FILE:ddceler-compare>
)

set_tests_properties(ddceler-validation PROPERTIES
  LABELS "dd4hep;validation"
  TIMEOUT 300  # 5 minutes
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Added DD4hep validation test: ddceler-validation")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  Set environment variables DDCELER_FILE1 and DDCELER_FILE2, then run:")
message(STATUS "    DDCELER_FILE1=/path/to/file1.root DDCELER_FILE2=/path/to/file2.root ctest -R ddceler-validation")
message(STATUS "")
message(STATUS "  Or use the executable directly:")
message(STATUS "    ddceler-compare file1.root file2.root")
message(STATUS "")
