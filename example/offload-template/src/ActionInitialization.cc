//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitialization::ActionInitialization() : G4VUserActionInitialization() {}

//---------------------------------------------------------------------------//
/*!
 * Set up all worker thread user actions.
 *
 * Celeritas is initialized and finalized automatically through Geant4 state
 * hooks, so no run action is needed for offloading.
 */
void ActionInitialization::Build() const
{
    // Initialize Geant4 user actions
    this->SetUserAction(new PrimaryGeneratorAction());

    // Print diagnostics
    this->SetUserAction(new EventAction());
}

}  // namespace example
}  // namespace celeritas
