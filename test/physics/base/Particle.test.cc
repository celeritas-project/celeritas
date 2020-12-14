//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Particle.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/ParticleTrackView.hh"

#include "celeritas_config.h"
#include "celeritas_test.hh"
#include "base/Array.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleStateStore.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/base/Units.hh"
#include "Particle.test.hh"

using celeritas::ParticleDef;
using celeritas::ParticleDefId;
using celeritas::ParticleParams;
using celeritas::ParticleParamsPointers;
using celeritas::ParticleStatePointers;
using celeritas::ParticleStateStore;
using celeritas::ParticleTrackView;

using celeritas::real_type;
using celeritas::ThreadId;
using celeritas::units::MevEnergy;

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS BASE
//---------------------------------------------------------------------------//

class ParticleTrackViewTest : public celeritas::Test
{
  protected:
    using Initializer_t = ParticleTrackView::Initializer_t;

    void SetUp() override
    {
        namespace pdg = celeritas::pdg;
        using celeritas::ParticleDef;
        using namespace celeritas::units;

        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Create particle defs, initialize on device
        ParticleParams::VecAnnotatedDefs defs;
        defs.push_back({{"electron", pdg::electron()},
                        {MevMass{0.5109989461}, ElementaryCharge{-1}, stable}});
        defs.push_back({{"gamma", pdg::gamma()}, {zero, zero, stable}});
        defs.push_back({{"neutron", PDGNumber{2112}},
                        {MevMass{939.565413}, zero, 1.0 / (879.4 * second)}});

        particle_params = std::make_shared<ParticleParams>(std::move(defs));
    }

    std::shared_ptr<ParticleParams> particle_params;
};

TEST_F(ParticleTrackViewTest, params_accessors)
{
    using celeritas::PDGNumber;
    const ParticleParams& defs = *this->particle_params;

    EXPECT_EQ(ParticleDefId(0), defs.find(PDGNumber(11)));
    EXPECT_EQ(ParticleDefId(1), defs.find(PDGNumber(22)));
    EXPECT_EQ(ParticleDefId(2), defs.find(PDGNumber(2112)));

    EXPECT_EQ(ParticleDefId(0), defs.find("electron"));
    EXPECT_EQ(ParticleDefId(1), defs.find("gamma"));
    EXPECT_EQ(ParticleDefId(2), defs.find("neutron"));

    ASSERT_EQ(3, defs.md().size());
    EXPECT_EQ("electron", defs.md()[0].name);
    EXPECT_EQ(PDGNumber(11), defs.md()[0].pdg_code);
}

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class ParticleTrackViewTestHost : public ParticleTrackViewTest
{
    using Base = ParticleTrackViewTest;

  protected:
    void SetUp() override
    {
        Base::SetUp();
        CHECK(particle_params);

        // Construct views
        params_view     = particle_params->host_pointers();
        state_view.vars = celeritas::make_span(state_storage);
    }

    celeritas::Array<celeritas::ParticleTrackState, 1> state_storage;

    ParticleParamsPointers params_view;
    ParticleStatePointers  state_view;
};

TEST_F(ParticleTrackViewTestHost, electron)
{
    ParticleTrackView particle(params_view, state_view, ThreadId(0));
    particle = Initializer_t{ParticleDefId{0}, MevEnergy{0.5}};

    EXPECT_DOUBLE_EQ(0.5, particle.energy().value());
    EXPECT_DOUBLE_EQ(0.5109989461, particle.mass().value());
    EXPECT_DOUBLE_EQ(-1., particle.charge().value());
    EXPECT_DOUBLE_EQ(0.0, particle.decay_constant());
    EXPECT_SOFT_EQ(0.86286196322132447, particle.speed().value());
    EXPECT_SOFT_EQ(25867950886.882648, celeritas::unit_cast(particle.speed()));
    EXPECT_SOFT_EQ(1.9784755992474248, particle.lorentz_factor());
    EXPECT_SOFT_EQ(0.87235253544653601, particle.momentum().value());
    EXPECT_SOFT_EQ(0.7609989461, particle.momentum_sq().value());
}

TEST_F(ParticleTrackViewTestHost, gamma)
{
    ParticleTrackView particle(params_view, state_view, ThreadId(0));
    particle = Initializer_t{ParticleDefId{1}, MevEnergy{10}};

    EXPECT_DOUBLE_EQ(0, particle.mass().value());
    EXPECT_DOUBLE_EQ(10, particle.energy().value());
    EXPECT_DOUBLE_EQ(1.0, particle.speed().value());
    EXPECT_DOUBLE_EQ(10, particle.momentum().value());
}

TEST_F(ParticleTrackViewTestHost, neutron)
{
    ParticleTrackView particle(params_view, state_view, ThreadId(0));
    particle = Initializer_t{ParticleDefId{2}, MevEnergy{20}};

    EXPECT_DOUBLE_EQ(20, particle.energy().value());
    EXPECT_DOUBLE_EQ(1.0 / 879.4, particle.decay_constant());
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class ParticleTrackViewTestDevice : public ParticleTrackViewTest
{
    using Base = ParticleTrackViewTest;
};

TEST_F(ParticleTrackViewTestDevice, calc_props)
{
    PTVTestInput input;
    input.init = {{ParticleDefId{0}, MevEnergy{0.5}},
                  {ParticleDefId{1}, MevEnergy{10}},
                  {ParticleDefId{2}, MevEnergy{20}}};

    ParticleStateStore pstates(input.init.size());
    input.params = particle_params->device_pointers();
    input.states = pstates.device_pointers();

    // Run GPU test
    auto result = ptv_test(input);

    // Check results
    // PRINT_EXPECTED(result.props);
    const double expected_props[] = {0.5,
                                     0.5109989461,
                                     -1,
                                     0,
                                     0.8628619632213,
                                     1.978475599247,
                                     0.8723525354465,
                                     0.7609989461,
                                     10,
                                     0,
                                     0,
                                     0,
                                     1,
                                     -1,
                                     10,
                                     100,
                                     20,
                                     939.565413,
                                     0,
                                     0.001137138958381,
                                     0.2031037086894,
                                     1.021286437031,
                                     194.8912941103,
                                     37982.61652};
    EXPECT_VEC_SOFT_EQ(expected_props, result.props);
}
#endif
