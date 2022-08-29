//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/PrimaryGenerator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PrimaryGeneratorTest : public celeritas_test::Test
{
  protected:
    void SetUp() override
    {
        namespace pdg = celeritas::pdg;

        constexpr auto zero = celeritas::zero_quantity();
        constexpr auto stable
            = celeritas::ParticleRecord::stable_decay_constant();

        // Create particle defs, initialize on device
        ParticleParams::Input defs;
        defs.push_back({"gamma", pdg::gamma(), zero, zero, stable});
        particles_ = std::make_shared<ParticleParams>(std::move(defs));
    }

    std::shared_ptr<ParticleParams> particles_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PrimaryGeneratorTest, host)
{
    PrimaryGeneratorOptions opts;
    EXPECT_FALSE(opts);

    opts.pdg = pdg::gamma();
    opts.energy = MevEnergy{10};
    opts.num_events = 2;
    opts.primaries_per_event = 3;

    PrimaryGenerator generate_primaries(particles_, opts);
    std::vector<int> event_id;
    std::vector<int> track_id;

    for (size_type i = 0; i < opts.num_events; ++i)
    {
        auto primaries = generate_primaries();
        EXPECT_EQ(opts.primaries_per_event, primaries.size());

        for (const auto& p : primaries)
        {
            EXPECT_EQ(ParticleId{0}, p.particle_id);
            EXPECT_EQ(MevEnergy{10}, p.energy);
            EXPECT_DOUBLE_EQ(0.0, p.time);
            EXPECT_VEC_SOFT_EQ(Real3({0, 0, 0}), p.position);
            EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), p.direction);
            event_id.push_back(p.event_id.unchecked_get());
            track_id.push_back(p.track_id.unchecked_get());
        }
    }
    auto primaries = generate_primaries();
    EXPECT_TRUE(primaries.empty());

    static const int expected_event_id[] = {0, 0, 0, 1, 1, 1};
    static const int expected_track_id[] = {0, 1, 2, 0, 1, 2};

    EXPECT_VEC_EQ(expected_event_id, event_id);
    EXPECT_VEC_EQ(expected_track_id, track_id);
}
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
