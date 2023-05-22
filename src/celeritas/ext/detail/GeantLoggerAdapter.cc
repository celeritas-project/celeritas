//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantLoggerAdapter.cc
//---------------------------------------------------------------------------//
#include "GeantLoggerAdapter.hh"

#include <cctype>
#include <memory>
#include <G4Version.hh>
#if CELER_G4SSBUF
#    include <G4strstreambuf.hh>
#else
#    include <G4ios.hh>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Redirect geant4's stdout/cerr on construction.
 */
GeantLoggerAdapter::GeantLoggerAdapter()
#if CELER_G4SSBUF
    : saved_cout_(G4coutbuf.GetDestination())
    , saved_cerr_(G4cerrbuf.GetDestination())
{
    G4coutbuf.SetDestination(this);
    G4cerrbuf.SetDestination(this);
}
#else
{
    // See Geant4 change global-V11-01-01
    G4iosSetDestination(this);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Restore iostream buffers on destruction.
 */
GeantLoggerAdapter::~GeantLoggerAdapter()
{
#if CELER_G4SSBUF
    if (G4coutbuf.GetDestination() == this)
    {
        G4coutbuf.SetDestination(saved_cout_);
    }
    if (G4cerrbuf.GetDestination() == this)
    {
        G4cerrbuf.SetDestination(saved_cerr_);
    }
#else
    G4iosSetDestination(nullptr);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Process stdout message.
 */
G4int GeantLoggerAdapter::ReceiveG4cout(G4String const& str)
{
    return this->log_impl(str, LogLevel::diagnostic);
}

//---------------------------------------------------------------------------//
/*!
 * Process stderr message.
 */
G4int GeantLoggerAdapter::ReceiveG4cerr(G4String const& str)
{
    return this->log_impl(str, LogLevel::info);
}

//---------------------------------------------------------------------------//
/*!
 * Log the actual message.
 */
G4int GeantLoggerAdapter::log_impl(G4String const& str, LogLevel level)
{
    // Output with dummy file/line
    ::celeritas::world_logger()({"Geant4", 0}, level) << trim(str);

    // 0 for success, -1 for failure
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
