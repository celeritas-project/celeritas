/*----------------------------------*-C-*------------------------------------*
 * Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*!
 * \file celeritas_version.h
 * \brief Version metadata for Celeritas.
 * \deprecated This file should be replaced by "corecel/Version.hh".
 */
//---------------------------------------------------------------------------//
#ifndef celeritas_version_h
#define celeritas_version_h

#if __cplusplus < 201703L
#    error "Celeritas requires C++17 or greater and is not C compatible"
#endif

#if __GNUC__ > 8 || __clang__
#    pragma GCC warning \
        "celeritas_version.h is deprecated and should be replaced by \"corecel/Version.hh\""
#endif

#include "corecel/Version.hh"

#endif /* celeritas_version_h */
