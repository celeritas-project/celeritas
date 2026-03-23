//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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
 * Count and offload Cherenkov/scintillation distributions across all streams.
 *
 * Call operator() at each G4 step (indexed by StreamId).
 * Call merged() on the master thread at end of run to aggregate.
 */
class DistOffloadCounter
{
  public:
    explicit DistOffloadCounter(StreamId::size_type num_streams)
        : counters_(num_streams)
    {
    }

    // Process a G4 step on the given stream
    void operator()(StreamId, G4Step const&);

    // Sum all per-stream counters (call only from master at end of run)
    StepCounters merged() const;

  private:
    std::vector<StepCounters> counters_;
    std::shared_ptr<GeantGeoParams const> geant_geo_;
    std::once_flag geant_geo_once_;
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
    UPStepAction make_stepping_action(StreamId) override;

    // Check counters at end-of-run on master
    void EndOfRunAction(G4Run const* run) override;

  private:
    std::optional<DistOffloadCounter> counter_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
