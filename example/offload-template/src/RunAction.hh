//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/RunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4UserRunAction.hh>

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas offloading interface.
 */
class RunAction : public G4UserRunAction
{
  public:
    // Construct empty
    RunAction();

    // Initialize Celeritas offloading interface
    void BeginOfRunAction(G4Run const* run) final;

    // Finalize Celeritas offloading interface
    void EndOfRunAction(G4Run const* run) final;
};
