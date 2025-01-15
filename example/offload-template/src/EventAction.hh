//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/EventAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Event.hh>
#include <G4UserEventAction.hh>

//---------------------------------------------------------------------------//
/*!
 * Initialize events in Celeritas.
 */
class EventAction : public G4UserEventAction
{
  public:
    // Construct empty
    EventAction();

    // Initialize event in Celeritas
    void BeginOfEventAction(G4Event const* event) final;

    // Flush any remaining particles to Celeritas
    void EndOfEventAction(G4Event const* event) final;
};
