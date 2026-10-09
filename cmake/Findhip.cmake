#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findhip
--------

Find the hip library.

#]=======================================================================]

# Suppress verbose cmake configuration output
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
find_package(hip QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hip CONFIG_MODE)

#-----------------------------------------------------------------------------#
