//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoInterface.cc
//---------------------------------------------------------------------------//
#include "GeoInterface.hh"

#include <iostream>

#include "corecel/io/LogHandlers.hh"

#include "GeoParamsInterface.hh"
#include "GeoTrackInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default virtual destructor
GeoParamsInterface::~GeoParamsInterface() = default;

//---------------------------------------------------------------------------//
//! Default virtual destructor
template<class RealType>
GeoTrackInterface<RealType>::~GeoTrackInterface() = default;

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
template class GeoTrackInterface<float>;
#endif
template class GeoTrackInterface<double>;

//---------------------------------------------------------------------------//
/*!
 * Local geometry logger: print on \em every thread, default "error" level.
 *
 * This is to print diagnostics about thread-local geometry issues.
 * Setting the "CELER_LOG_GEO" environment variable to "debug", "info",
 * "error", etc. will change the default log level.
 *
 * \sa CELER_LOG .
 */
Logger& geo_logger()
{
    static Logger logger{StreamLogHandler{std::clog},
                         getenv_loglevel("CELER_LOG_GEO", LogLevel::error)};
    return logger;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
