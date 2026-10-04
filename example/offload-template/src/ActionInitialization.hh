//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Initialize all user action classes.
 *
 * Celeritas offloading setup and teardown are driven automatically by Geant4
 * state hooks, so no user run action is required.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    // Construct empty
    ActionInitialization();

    // Worker thread actions
    void Build() const final;
};
}  // namespace example
}  // namespace celeritas
