//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantLoggerAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4String.hh>
#include <G4Types.hh>
#include <G4Version.hh>
#include <G4coutDestination.hh>
#if G4VERSION_NUMBER > 1111
// No ability to include G4strstreambuf
#    define CELER_G4SSBUF 0
#else
#    define CELER_G4SSBUF 1
#endif

#include "corecel/io/LoggerTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Handle log messages from Geant4 while in scope.
 */
class GeantLoggerAdapter : public G4coutDestination
{
  public:
    // Assign to Geant handlers on construction
    GeantLoggerAdapter();
    ~GeantLoggerAdapter();

    // Handle error messages
    G4int ReceiveG4cout(G4String const& str) final;
    G4int ReceiveG4cerr(G4String const& str) final;

  private:
    //// DATA ////

#if CELER_G4SSBUF
    G4coutDestination* saved_cout_;
    G4coutDestination* saved_cerr_;
#endif

    //// IMPLEMENTATION ////
    G4int log_impl(G4String const& str, LogLevel level);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
