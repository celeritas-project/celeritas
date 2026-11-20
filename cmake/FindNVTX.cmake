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

Result variables
^^^^^^^^^^^^^^^^

``NVTX_INCLUDE_DIRS``
  Include directory containing ``nvtx3/nvToolsExt.h``.

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
set(NVTX_INCLUDE_DIRS "${NVTX_INCLUDE_DIR}")

#-----------------------------------------------------------------------------#
