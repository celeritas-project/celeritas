#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findnlohmann_json
-----------------

Find the "JSON for modern c++" library.

#]=======================================================================]

find_package(nlohmann_json QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nlohmann_json CONFIG_MODE)

#-----------------------------------------------------------------------------#
