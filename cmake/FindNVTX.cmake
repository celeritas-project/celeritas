#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindNVTX
--------

Find the header-only NVTX v3 profiling annotation library. The headers are
part of the CUDA toolkit but may be packaged separately, so the toolkit include
directories are searched first and then the default search paths (including
``NVTX_ROOT``). Set ``NVTX_INCLUDE_DIR`` to override the search.

Result targets
^^^^^^^^^^^^^^

``NVTX::nvtx3``
  Interface target providing the NVTX include directory.

#]=======================================================================]

# Prefer the copy in the CUDA toolkit to one in another prefix
find_path(NVTX_INCLUDE_DIR
  NAMES nvtx3/nvToolsExt.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  NO_DEFAULT_PATH
)
find_path(NVTX_INCLUDE_DIR
  NAMES nvtx3/nvToolsExt.h
)
mark_as_advanced(NVTX_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX
  REQUIRED_VARS NVTX_INCLUDE_DIR
  REASON_FAILURE_MESSAGE
    "install the development headers for the CUDA NVTX component"
)

if(NVTX_FOUND AND NOT TARGET NVTX::nvtx3)
  set(NVTX_INCLUDE_DIRS "${NVTX_INCLUDE_DIR}")
  add_library(NVTX::nvtx3 INTERFACE IMPORTED)
  set_target_properties(NVTX::nvtx3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVTX_INCLUDE_DIR}"
    # NVTX loads the profiler injection library with dlopen
    INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS}"
  )
endif()

#-----------------------------------------------------------------------------#
