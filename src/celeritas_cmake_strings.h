/*----------------------------------*-C-*------------------------------------*
 * Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*!
 * \file celeritas_cmake_strings.h
 * \brief Detailed CMake configuration information.
 * \deprecated This file should be replaced by "corecel/Config.hh".
 */
//---------------------------------------------------------------------------//
#ifndef celeritas_cmake_strings_h
#define celeritas_cmake_strings_h

#if __cplusplus < 201703L
#    error "Celeritas requires C++17 or greater and is not C compatible"
#endif

#if __GNUC__ > 8 || __clang__
#    pragma GCC warning \
        "celeritas_cmake_strings.h is deprecated and should be replaced by \"corecel/Config.hh\""
#endif

#include "corecel/Config.hh"

#endif /* celeritas_cmake_strings_h */
