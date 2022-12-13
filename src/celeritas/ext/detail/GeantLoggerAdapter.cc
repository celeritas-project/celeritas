//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantLoggerAdapter.cc
//---------------------------------------------------------------------------//
#include "GeantLoggerAdapter.hh"

#include <cctype>
#include <G4strstreambuf.hh>

#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Redirect geant4's stdout/cerr on construction.
 */
GeantLoggerAdapter::GeantLoggerAdapter()
    : saved_cout_(G4coutbuf.GetDestination())
    , saved_cerr_(G4cerrbuf.GetDestination())
{
    G4coutbuf.SetDestination(this);
    G4cerrbuf.SetDestination(this);
}

//---------------------------------------------------------------------------//
/*!
 * Restore iostream buffers on destruction.
 */
GeantLoggerAdapter::~GeantLoggerAdapter()
{
    if (G4coutbuf.GetDestination() == this)
    {
        G4coutbuf.SetDestination(saved_cout_);
    }
    if (G4cerrbuf.GetDestination() == this)
    {
        G4cerrbuf.SetDestination(saved_cerr_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process stdout message.
 */
G4int GeantLoggerAdapter::ReceiveG4cout(const G4String& str)
{
    return this->log_impl(str, LogLevel::diagnostic);
}

//---------------------------------------------------------------------------//
/*!
 * Process stderr message.
 */
G4int GeantLoggerAdapter::ReceiveG4cerr(const G4String& str)
{
    return this->log_impl(str, LogLevel::info);
}

//---------------------------------------------------------------------------//
/*!
 * Log the actual message.
 */
G4int GeantLoggerAdapter::log_impl(const G4String& str, LogLevel level)
{
    G4String temp(str);
    // Strip trailing whitespace
    while (!temp.empty()
           && std::isspace(static_cast<unsigned char>(temp.back())))
    {
        temp.pop_back();
    }

    // Output with dummy file/line
    ::celeritas::world_logger()({"Geant4", 0}, level) << temp;

    // 0 for success, -1 for failure
    return 0;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
