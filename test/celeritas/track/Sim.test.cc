//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/Sim.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/StateDataStore.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/SimTrackView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SimTest : public GeantTestBase
{
  protected:
    using SimStateStore = StateDataStore<SimStateData, MemSpace::host>;
    using ParticleStateStore
        = StateDataStore<ParticleStateData, MemSpace::host>;
    using MevEnergy = units::MevEnergy;

  protected:
    std::string_view gdml_basename() const override
    {
        return "four-steel-slabs"sv;
    }

    void SetUp() override
    {
        // Allocate particle and sim states
        auto state_size = 1;
        particle_state_
            = ParticleStateStore(this->particle()->host_ref(), state_size);
        sim_state_ = SimStateStore(this->sim()->host_ref(), state_size);
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    // Initialize a track
    ParticleTrackView make_par_view(PDGNumber pdg, MevEnergy energy)
    {
        CELER_EXPECT(pdg);
        CELER_EXPECT(energy > zero_quantity());
        auto pid = this->particle()->find(pdg);
        CELER_ASSERT(pid);

        ParticleTrackView par{this->particle()->host_ref(),
                              particle_state_.ref(),
                              TrackSlotId{0}};
        ParticleTrackView::Initializer_t init;
        init.particle_id = pid;
        init.energy = energy;
        par = init;
        return par;
    }

    ParticleStateStore particle_state_;
    SimStateStore sim_state_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SimTest, looping)
{
    LoopingThreshold expected;
    expected.threshold_energy = MevEnergy{250};
    expected.max_steps = 1000;
    expected.max_subthreshold_steps = 100;

    // Check looping threshold parameters for each particle
    SimTrackView sim(this->sim()->host_ref(), sim_state_.ref(), TrackSlotId{0});
    sim = SimTrackInitializer{
        TrackId{0}, TrackId{}, PrimaryId{}, EventId{0}, 0};
    for (auto pid : range(ParticleId{this->particle()->size()}))
    {
        auto const& looping = sim.looping_threshold(pid);
        EXPECT_EQ(expected.threshold_energy, looping.threshold_energy);
        EXPECT_EQ(expected.max_steps, looping.max_steps);
        EXPECT_EQ(expected.max_subthreshold_steps,
                  looping.max_subthreshold_steps);
    }

    MevEnergy const eps{1e-6};
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Below the threshold energy
        auto par = this->make_par_view(pdg::electron(),
                                       expected.threshold_energy - eps);

        // Step up to just under the threshold number iterations while looping
        // for low energy particles
        for (size_type i = 0; i < expected.max_subthreshold_steps - 1; ++i)
        {
            sim.update_looping(/* is looping in field propagator = */ true);
        }
        EXPECT_EQ(expected.max_subthreshold_steps - 1, sim.num_looping_steps());
        EXPECT_FALSE(sim.is_looping(par.particle_id(), par.energy()));

        // Take one more step while looping
        sim.update_looping(/* is looping in field propagator = */ true);
        EXPECT_EQ(expected.max_subthreshold_steps, sim.num_looping_steps());
        EXPECT_TRUE(sim.is_looping(par.particle_id(), par.energy()));
    }
    {
        // Above the threshold energy
        auto par = this->make_par_view(pdg::electron(),
                                       expected.threshold_energy + eps);
        EXPECT_FALSE(sim.is_looping(par.particle_id(), par.energy()));

        // Reset the looping step count and step up to just under the threshold
        // number iterations while looping
        sim.update_looping(/* is looping in field propagator = */ false);
        for (size_type i = 0; i < expected.max_steps - 1; ++i)
        {
            sim.update_looping(/* is looping in field propagator = */ true);
        }
        EXPECT_EQ(expected.max_steps - 1, sim.num_looping_steps());
        EXPECT_FALSE(sim.is_looping(par.particle_id(), par.energy()));

        // Take one more step while looping
        sim.update_looping(/* is looping in field propagator = */ true);
        EXPECT_EQ(expected.max_steps, sim.num_looping_steps());
        EXPECT_TRUE(sim.is_looping(par.particle_id(), par.energy()));
    }
}

TEST_F(SimTest, weight)
{
    SimTrackView sim(this->sim()->host_ref(), sim_state_.ref(), TrackSlotId{0});
    SimTrackInitializer init;
    init.track_id = TrackId{0};
    init.parent_id = TrackId{};
    init.event_id = EventId{0};
    init.time = 0.0;

    real_type expected_weight = 0.5;
    init.weight = expected_weight;
    sim = init;

    // Check if weight was correctly stored
    EXPECT_EQ(expected_weight, sim.weight());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
