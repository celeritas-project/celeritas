//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Logger.hh"

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * \def CELER_LOG_GEO
 *
 * Like \c CELER_LOG but for geometry-related issues that are likely only
 * happen on a single process or thread. Use \em very sparingly due to
 * erbosity. This should be used for geometry-related error messages from an
 * event or track at runtime.
 */
#define CELER_LOG_GEO(LEVEL) \
    ::celeritas::geo_logger()(CELER_CODE_PROVENANCE, \
                              ::celeritas::LogLevel::LEVEL)

// Allow CELER_LOG_GEO to be present (but ignored) in device code
#if CELER_DEVICE_COMPILE
#    undef CELER_LOG_GEO
#    define CELER_LOG_GEO(LEVEL) ::celeritas::null_log_message()
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//

// Local geometry logger (print on *every* process using CELER_LOG_GEO)
Logger& geo_logger();

//---------------------------------------------------------------------------//
}  // namespace celeritas
