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
#include "base/CollectionStateStore.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/Units.hh"
#include "io/ImportData.hh"
#include "io/RootImporter.hh"
#include "Particle.test.hh"

using celeritas::ParticleId;
using celeritas::ParticleParams;
using celeritas::ParticleTrackView;

using celeritas::ImportData;
using celeritas::RootImporter;

using celeritas::real_type;
using celeritas::ThreadId;
using celeritas::units::MevEnergy;

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS BASE
//---------------------------------------------------------------------------//

class ParticleTest : public celeritas::Test
{
  protected:
    using Initializer_t = ParticleTrackView::Initializer_t;

    void SetUp() override
    {
        namespace pdg = celeritas::pdg;
        using namespace celeritas::units;

        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = celeritas::ParticleDef::stable_decay_constant();

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
    using celeritas::PDGNumber;
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

class ParticleImportTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root("geant4_data", "ImportData");
        ;
    }
    std::string root_filename_;
    ImportData  data_;
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

    // Particle ordering is the same as in the ROOT file
    // clang-format off
    const std::string expected_loaded_names[] = {"gamma", "e-", "e+", "mu-",
        "mu+", "pi+", "pi-", "kaon+", "kaon-", "proton", "anti_proton",
        "deuteron", "anti_deuteron", "He3", "anti_He3", "triton",
        "anti_triton", "alpha", "anti_alpha"};
    const int expected_loaded_pdgs[] = {22, 11, -11, 13, -13, 211, -211, 321,
        -321, 2212, -2212, 1000010020, -1000010020, 1000020030, -1000020030,
        1000010030, -1000010030, 1000020040, -1000020040};
    // clang-format on

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
        resize(&state_value, particle_params->host_pointers(), 1);
        state_ref = state_value;
    }

    ParticleStateData<Ownership::value, MemSpace::host>     state_value;
    ParticleStateData<Ownership::reference, MemSpace::host> state_ref;
};

TEST_F(ParticleTestHost, electron)
{
    ParticleTrackView particle(
        particle_params->host_pointers(), state_ref, ThreadId(0));
    particle = Initializer_t{ParticleId{0}, MevEnergy{0.5}};

    EXPECT_DOUBLE_EQ(0.5, particle.energy().value());
    EXPECT_DOUBLE_EQ(0.5109989461, particle.mass().value());
    EXPECT_DOUBLE_EQ(-1., particle.charge().value());
    EXPECT_DOUBLE_EQ(0.0, particle.decay_constant());
    EXPECT_SOFT_EQ(0.86286196322132447, particle.speed().value());
    EXPECT_SOFT_EQ(25867950886.882648, celeritas::unit_cast(particle.speed()));
    EXPECT_SOFT_EQ(1.9784755992474248, particle.lorentz_factor());
    EXPECT_SOFT_EQ(0.87235253544653601, particle.momentum().value());
    EXPECT_SOFT_EQ(0.7609989461, particle.momentum_sq().value());

    // Stop the particle
    EXPECT_FALSE(particle.is_stopped());
    particle.energy(zero_quantity());
    EXPECT_TRUE(particle.is_stopped());
}

TEST_F(ParticleTestHost, gamma)
{
    ParticleTrackView particle(
        particle_params->host_pointers(), state_ref, ThreadId(0));
    particle = Initializer_t{ParticleId{1}, MevEnergy{10}};

    EXPECT_DOUBLE_EQ(0, particle.mass().value());
    EXPECT_DOUBLE_EQ(10, particle.energy().value());
    EXPECT_DOUBLE_EQ(1.0, particle.speed().value());
    EXPECT_DOUBLE_EQ(10, particle.momentum().value());
}

TEST_F(ParticleTestHost, neutron)
{
    ParticleTrackView particle(
        particle_params->host_pointers(), state_ref, ThreadId(0));
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

TEST_F(ParticleDeviceTest, TEST_IF_CELERITAS_CUDA(calc_props))
{
    PTVTestInput input;
    input.init = {{ParticleId{0}, MevEnergy{0.5}},
                  {ParticleId{1}, MevEnergy{10}},
                  {ParticleId{2}, MevEnergy{20}}};

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init.size());
    input.params = particle_params->device_pointers();
    input.states = pstates.ref();

    // Run GPU test
    PTVTestOutput result;
#if CELERITAS_USE_CUDA
    result = ptv_test(input);
#endif

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
