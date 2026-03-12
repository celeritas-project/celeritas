//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <memory>
#include <G4UserSteppingAction.hh>

#include "corecel/Assert.hh"
#include "geocel/GeantGeoParams.hh"

#include "IntegrationTestBase.hh"

namespace celeritas
{
namespace test
{
struct StepCounters
{
    std::uint64_t optical{0};
    std::uint64_t other{0};
};

//---------------------------------------------------------------------------//
/*!
 * Offload Cherenkov and scintillation tracks at every step.
 */
class DistOffloadSteppingAction final : public G4UserSteppingAction
{
  public:
    using SPCounters = std::shared_ptr<StepCounters>;

    // Construct with thread-local counter reference.
    explicit DistOffloadSteppingAction(SPCounters counters)
        : counters_{counters}
    {
        CELER_EXPECT(counters);
    }

    void UserSteppingAction(G4Step const*) final;

  private:
    SPCounters counters_;
    std::shared_ptr<GeantGeoParams const> geant_geo_;
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
    UPStepAction make_stepping_action() override;

    // Check counters at end-of-run on master
    void EndOfRunAction(G4Run const* run) override;

    // Calculate and return sum across all threads
    // NOT thread safe (do only in end of run for master)
    StepCounters merge_step_counters() const;

  private:
    std::vector<DistOffloadSteppingAction::SPCounters> counters_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
