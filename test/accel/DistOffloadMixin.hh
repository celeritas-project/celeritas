//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <memory>

#include "IntegrationTestBase.hh"

namespace celeritas
{
class GeantGeoParams;
namespace test
{
struct StepCounters
{
    std::uint64_t optical{0};
    std::uint64_t other{0};
};

//---------------------------------------------------------------------------//
/*!
 * Set up to offload optical distributions.
 */
class DistOffloadMixin : virtual public IntegrationTestBase
{
  public:
    PhysicsInput make_physics_input() const override;
    SetupOptions make_setup_options() const override;
    FuncLocalStep make_step_callback() override;

    // Check counters at end-of-run on master
    void BeginOfRunAction(G4Run const* run) override;
    void EndOfRunAction(G4Run const* run) override;

  private:
    std::vector<StepCounters> counters_;
    std::shared_ptr<GeantGeoParams const> geant_geo_;

    // Process a G4 step on the given stream
    void step(StreamId, G4Step const&);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
