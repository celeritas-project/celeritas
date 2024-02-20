//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/PrimaryGenerator.hh"

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#if CELERITAS_USE_JSON
#    include "celeritas/phys/PrimaryGeneratorOptionsIO.json.hh"
#endif

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

class PrimaryGeneratorTest : public Test
{
  protected:
    void SetUp() override
    {
        using namespace constants;
        constexpr auto zero = zero_quantity();

        // Create particle defs
        ParticleParams::Input defs{
            {"gamma", pdg::gamma(), zero, zero, stable_decay_constant},
            {"electron",
             pdg::electron(),
             units::MevMass{0.5109989461},
             units::ElementaryCharge{-1},
             stable_decay_constant}};
        particles_ = std::make_shared<ParticleParams>(std::move(defs));
    }

    std::shared_ptr<ParticleParams> particles_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PrimaryGeneratorTest, basic)
{
    PrimaryGenerator::Input inp;
    inp.pdg = {pdg::gamma(), pdg::electron()};
    inp.num_events = 2;
    inp.primaries_per_event = 3;
    inp.sample_energy = DeltaDistribution<real_type>(10);
    inp.sample_pos = DeltaDistribution<Real3>(Real3{1, 2, 3});
    inp.sample_dir = IsotropicDistribution<real_type>();
    PrimaryGenerator generate_primaries(particles_, inp);
    EXPECT_EQ(2, generate_primaries.num_events());

    std::vector<int> particle_id;
    std::vector<int> event_id;
    std::vector<int> track_id;

    for (size_type i = 0; i < inp.num_events; ++i)
    {
        auto primaries = generate_primaries();
        EXPECT_EQ(inp.primaries_per_event, primaries.size());

        for (auto const& p : primaries)
        {
            EXPECT_EQ(MevEnergy{10}, p.energy);
            EXPECT_REAL_EQ(0.0, p.time);
            EXPECT_VEC_SOFT_EQ(Real3({1, 2, 3}), p.position);
            EXPECT_TRUE(is_soft_unit_vector(p.direction));
            particle_id.push_back(p.particle_id.unchecked_get());
            event_id.push_back(p.event_id.unchecked_get());
            track_id.push_back(p.track_id.unchecked_get());
        }
    }
    auto primaries = generate_primaries();
    EXPECT_TRUE(primaries.empty());

    static int const expected_particle_id[] = {0, 1, 0, 0, 1, 0};
    static int const expected_event_id[] = {0, 0, 0, 1, 1, 1};
    static int const expected_track_id[] = {0, 1, 2, 0, 1, 2};

    EXPECT_VEC_EQ(expected_particle_id, particle_id);
    EXPECT_VEC_EQ(expected_event_id, event_id);
    EXPECT_VEC_EQ(expected_track_id, track_id);
}

TEST_F(PrimaryGeneratorTest, options)
{
    using DS = DistributionSelection;

    PrimaryGeneratorOptions opts;
    opts.pdg = {pdg::gamma()};
    opts.num_events = 1;
    opts.primaries_per_event = 10;
    opts.energy = {DS::delta, {1}};
    opts.position = {DS::box, {-3, -3, -3, 3, 3, 3}};
    opts.direction = {DS::isotropic, {}};

    auto generate_primaries = PrimaryGenerator::from_options(particles_, opts);
    EXPECT_EQ(1, generate_primaries.num_events());
    auto primaries = generate_primaries();
    EXPECT_EQ(10, primaries.size());

    for (size_type i : range(primaries.size()))
    {
        auto const& p = primaries[i];
        EXPECT_EQ(ParticleId{0}, p.particle_id);
        EXPECT_EQ(TrackId{i}, p.track_id);
        EXPECT_EQ(EventId{0}, p.event_id);
        EXPECT_EQ(MevEnergy{1}, p.energy);
        EXPECT_REAL_EQ(0.0, p.time);
        for (auto x : p.position)
        {
            EXPECT_TRUE(x >= -3 && x <= 3);
        }
        EXPECT_TRUE(is_soft_unit_vector(p.direction));
    }
    primaries = generate_primaries();
    EXPECT_TRUE(primaries.empty());

#if CELERITAS_USE_JSON
    {
        nlohmann::json out = opts;
        EXPECT_JSON_EQ(
            R"json({"direction":{"distribution":"isotropic","params":[]},"energy":{"distribution":"delta","params":[1.0]},"num_events":1,"pdg":[22],"position":{"distribution":"box","params":[-3.0,-3.0,-3.0,3.0,3.0,3.0]},"primaries_per_event":10,"seed":0})json",
            std::string(out.dump()));
    }
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
