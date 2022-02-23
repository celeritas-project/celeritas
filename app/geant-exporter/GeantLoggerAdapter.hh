//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantLoggerAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4coutDestination.hh>

#include "comm/LoggerTypes.hh"

namespace celeritas
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
    G4int ReceiveG4cout(const G4String& str) final;
    G4int ReceiveG4cerr(const G4String& str) final;

  private:
    //// DATA ////

    G4coutDestination* saved_cout_;
    G4coutDestination* saved_cerr_;

    //// IMPLEMENTATION ////
    G4int log_impl(const G4String& str, LogLevel level);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
