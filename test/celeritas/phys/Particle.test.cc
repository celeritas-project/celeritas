//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Particle.test.cc
//---------------------------------------------------------------------------//
#include "Particle.test.hh"

#include "celeritas_config.h"
#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/detail/ScopedRootErrorHandler.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS BASE
//---------------------------------------------------------------------------//

class ParticleTest : public Test
{
  protected:
    using Initializer_t = ParticleTrackView::Initializer_t;
    using MevEnergy     = units::MevEnergy;

    void SetUp() override
    {
        using namespace units;

        constexpr auto zero   = zero_quantity();
        constexpr auto stable = ParticleRecord::stable_decay_constant();

        // Create particle defs, initialize on device
        ParticleParams::Input defs;
        defs.push_back({"electron",
                        pdg::electron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{-1},
                        stable});
        defs.push_back({"gamma", pdg::gamma(), zero, zero, stable});
        defs.push_back({"neutron",
                        PDGNumber{2112},
                        MevMass{939.565413},
                        zero,
                        1.0 / (879.4 * second)});

        particle_params = std::make_shared<ParticleParams>(std::move(defs));
    }

    std::shared_ptr<ParticleParams> particle_params;
};

TEST_F(ParticleTest, params_accessors)
{
    const ParticleParams& defs = *this->particle_params;

    EXPECT_EQ(ParticleId(0), defs.find(PDGNumber(11)));
    EXPECT_EQ(ParticleId(1), defs.find(PDGNumber(22)));
    EXPECT_EQ(ParticleId(2), defs.find(PDGNumber(2112)));

    EXPECT_EQ(ParticleId(0), defs.find("electron"));
    EXPECT_EQ(ParticleId(1), defs.find("gamma"));
    EXPECT_EQ(ParticleId(2), defs.find("neutron"));

    EXPECT_EQ("electron", defs.id_to_label(ParticleId(0)));
    EXPECT_EQ(PDGNumber(11), defs.id_to_pdg(ParticleId(0)));
}

//---------------------------------------------------------------------------//
// IMPORT PARTICLE DATA TEST
//---------------------------------------------------------------------------//

class ParticleImportTest : public Test
{
  protected:
    void SetUp() override
    {
        root_filename_
            = this->test_data_path("celeritas", "four-steel-slabs.root");
        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root();
    }
    std::string root_filename_;
    ImportData  data_;

    detail::ScopedRootErrorHandler scoped_root_error_;
};

TEST_F(ParticleImportTest, TEST_IF_CELERITAS_USE_ROOT(import_particle))
{
    const auto particles = ParticleParams::from_import(data_);

    // Check electron data
    ParticleId electron_id = particles->find(PDGNumber(11));
    ASSERT_GE(electron_id.get(), 0);
    const auto& electron = particles->get(electron_id);
    EXPECT_SOFT_EQ(0.510998910, electron.mass().value());
    EXPECT_EQ(-1, electron.charge().value());
    EXPECT_EQ(0, electron.decay_constant());
    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (auto particle_id : range(ParticleId{particles->size()}))
    {
        loaded_names.push_back(particles->id_to_label(particle_id));
        loaded_pdgs.push_back(particles->id_to_pdg(particle_id).get());
    }

    const std::string expected_loaded_names[] = {"gamma", "e-", "e+"};
    const int         expected_loaded_pdgs[]  = {22, 11, -11};

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class ParticleTestHost : public ParticleTest
{
    using Base = ParticleTest;

  protected:
    void SetUp() override
    {
        Base::SetUp();
        CELER_ASSERT(particle_params);

        // Construct views
        resize(&state_value, particle_params->host_ref(), 1);
        state_ref = state_value;
    }

    HostVal<ParticleStateData> state_value;
    HostRef<ParticleStateData> state_ref;
};

TEST_F(ParticleTestHost, electron)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, ThreadId(0));
    particle = Initializer_t{ParticleId{0}, MevEnergy{0.5}};

    EXPECT_DOUBLE_EQ(0.5, particle.energy().value());
    EXPECT_DOUBLE_EQ(0.5109989461, particle.mass().value());
    EXPECT_DOUBLE_EQ(-1., particle.charge().value());
    EXPECT_DOUBLE_EQ(0.0, particle.decay_constant());
    EXPECT_SOFT_EQ(0.74453076757415848, particle.beta_sq());
    EXPECT_SOFT_EQ(0.86286196322132447, particle.speed().value());
    EXPECT_SOFT_EQ(25867950886.882648, native_value_from(particle.speed()));
    EXPECT_SOFT_EQ(1.9784755992474248, particle.lorentz_factor());
    EXPECT_SOFT_EQ(0.87235253544653601, particle.momentum().value());
    EXPECT_SOFT_EQ(0.7609989461, particle.momentum_sq().value());

    // Stop the particle
    EXPECT_FALSE(particle.is_stopped());
    particle.subtract_energy(MevEnergy{0.25});
    EXPECT_DOUBLE_EQ(0.25, particle.energy().value());
    particle.energy(zero_quantity());
    EXPECT_TRUE(particle.is_stopped());
    EXPECT_DOUBLE_EQ(0.0, particle.energy().value());
}

TEST_F(ParticleTestHost, gamma)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, ThreadId(0));
    particle = Initializer_t{ParticleId{1}, MevEnergy{10}};

    EXPECT_DOUBLE_EQ(0, particle.mass().value());
    EXPECT_DOUBLE_EQ(10, particle.energy().value());
    EXPECT_DOUBLE_EQ(1.0, particle.beta_sq());
    EXPECT_DOUBLE_EQ(1.0, particle.speed().value());
    EXPECT_DOUBLE_EQ(10, particle.momentum().value());
}

TEST_F(ParticleTestHost, neutron)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, ThreadId(0));
    particle = Initializer_t{ParticleId{2}, MevEnergy{20}};

    EXPECT_DOUBLE_EQ(20, particle.energy().value());
    EXPECT_DOUBLE_EQ(1.0 / 879.4, particle.decay_constant());
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class ParticleDeviceTest : public ParticleTest
{
    using Base = ParticleTest;
};

TEST_F(ParticleDeviceTest, TEST_IF_CELER_DEVICE(calc_props))
{
    PTVTestInput input;
    input.init = {{ParticleId{0}, MevEnergy{0.5}},
                  {ParticleId{1}, MevEnergy{10}},
                  {ParticleId{2}, MevEnergy{20}}};

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        particle_params->host_ref(), input.init.size());
    input.params = particle_params->device_ref();
    input.states = pstates.ref();

    // Run GPU test
    PTVTestOutput result;
    result = ptv_test(input);

    // Check results
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
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
