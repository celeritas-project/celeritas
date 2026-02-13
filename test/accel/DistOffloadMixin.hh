//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserSteppingAction.hh>

#include "IntegrationTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Offload Cherenkov and scintillation tracks at every step.
 */
class DistOffloadSteppingAction final : public G4UserSteppingAction
{
  public:
    void UserSteppingAction(G4Step const*) final;
};

//---------------------------------------------------------------------------//
/*!
 * Set up to offload optical distributions.
 */
class DistOffloadMixin : virtual public IntegrationTestBase
{
  public:
    PhysicsInput make_physics_input() const override;
    SetupOptions make_setup_options() override;
    UPStepAction make_stepping_action() override
    {
        return std::make_unique<DistOffloadSteppingAction>();
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
