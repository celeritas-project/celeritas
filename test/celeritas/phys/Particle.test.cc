//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Particle.test.cc
//---------------------------------------------------------------------------//
#include "Particle.test.hh"

#include "corecel/Config.hh"

#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleParamsOutput.hh"
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
    using MevEnergy = units::MevEnergy;

    void SetUp() override
    {
        using namespace constants;
        using namespace units;

        constexpr auto zero = zero_quantity();

        // Create particle defs, initialize on device
        ParticleParams::Input defs;
        defs.push_back({"electron",
                        pdg::electron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{-1},
                        stable_decay_constant});
        defs.push_back(
            {"gamma", pdg::gamma(), zero, zero, stable_decay_constant});
        defs.push_back({"neutron",
                        PDGNumber{2112},
                        MevMass{939.565413},
                        zero,
                        1.0 / (879.4 * second)});
        defs.push_back({"positron",
                        pdg::positron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{1},
                        stable_decay_constant});

        particle_params = std::make_shared<ParticleParams>(std::move(defs));
    }

    std::shared_ptr<ParticleParams> particle_params;
};

TEST_F(ParticleTest, params_accessors)
{
    ParticleParams const& defs = *this->particle_params;

    EXPECT_EQ(ParticleId(0), defs.find(PDGNumber(11)));
    EXPECT_EQ(ParticleId(1), defs.find(PDGNumber(22)));
    EXPECT_EQ(ParticleId(2), defs.find(PDGNumber(2112)));
    EXPECT_EQ(ParticleId(3), defs.find(PDGNumber(-11)));

    EXPECT_EQ(ParticleId(0), defs.find("electron"));
    EXPECT_EQ(ParticleId(1), defs.find("gamma"));
    EXPECT_EQ(ParticleId(2), defs.find("neutron"));
    EXPECT_EQ(ParticleId(3), defs.find("positron"));

    EXPECT_EQ("electron", defs.id_to_label(ParticleId(0)));
    EXPECT_EQ(PDGNumber(11), defs.id_to_pdg(ParticleId(0)));
}

TEST_F(ParticleTest, output)
{
    ParticleParamsOutput out(this->particle_params);
    EXPECT_EQ("particle", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"particle","_units":{"charge":"e","mass":"MeV/c^2"},"charge":[-1.0,0.0,0.0,1.0],"decay_constant":[0.0,0.0,0.0011371389583807142,0.0],"is_antiparticle":[false,false,false,true],"label":["electron","gamma","neutron","positron"],"mass":[0.5109989461,0.0,939.565413,0.5109989461],"pdg":[11,22,2112,-11]})json",
            to_string(out));
    }
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
    ImportData data_;

    ScopedRootErrorHandler scoped_root_error_;
};

TEST_F(ParticleImportTest, TEST_IF_CELERITAS_USE_ROOT(import_particle))
{
    auto const particles = ParticleParams::from_import(data_);

    // Check electron data
    ParticleId electron_id = particles->find(PDGNumber(11));
    ASSERT_GE(electron_id.get(), 0);
    auto const& electron = particles->get(electron_id);
    EXPECT_SOFT_EQ(0.510998910, electron.mass().value());
    EXPECT_EQ(-1, electron.charge().value());
    EXPECT_EQ(0, electron.decay_constant());
    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int> loaded_pdgs;
    for (auto particle_id : range(ParticleId{particles->size()}))
    {
        loaded_names.push_back(particles->id_to_label(particle_id));
        loaded_pdgs.push_back(particles->id_to_pdg(particle_id).get());
    }

    std::string const expected_loaded_names[]
        = {"gamma", "e-", "e+", "mu-", "mu+"};
    int const expected_loaded_pdgs[] = {22, 11, -11, 13, -13};

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
        particle_params->host_ref(), state_ref, TrackSlotId(0));
    particle = Initializer_t{ParticleId{0}, MevEnergy{0.5}};

    EXPECT_REAL_EQ(0.5, particle.energy().value());
    EXPECT_REAL_EQ(0.5109989461, particle.mass().value());
    EXPECT_REAL_EQ(-1., particle.charge().value());
    EXPECT_REAL_EQ(0.0, particle.decay_constant());
    EXPECT_FALSE(particle.is_antiparticle());
    EXPECT_TRUE(particle.is_stable());
    EXPECT_FALSE(particle.is_heavy());
    EXPECT_REAL_EQ(1.0109989461, particle.total_energy().value());
    EXPECT_SOFT_EQ(0.74453076757415848, particle.beta_sq());
    EXPECT_SOFT_EQ(0.86286196322132447,
                   value_as<units::LightSpeed>(particle.speed()));
    EXPECT_SOFT_EQ(1.9784755992474248, particle.lorentz_factor());
    EXPECT_SOFT_EQ(0.87235253544653601, particle.momentum().value());
    EXPECT_SOFT_EQ(0.7609989461, particle.momentum_sq().value());

    // Stop the particle
    EXPECT_FALSE(particle.is_stopped());
    particle.subtract_energy(MevEnergy{0.25});
    EXPECT_REAL_EQ(0.25, particle.energy().value());
    particle.energy(zero_quantity());
    EXPECT_TRUE(particle.is_stopped());
    EXPECT_REAL_EQ(0.0, particle.energy().value());

    {
        SCOPED_TRACE("High energy electron");
        particle = Initializer_t{ParticleId{0}, MevEnergy{1e12}};
        EXPECT_LE(particle.speed().value(), 1);
        EXPECT_LE(particle.beta_sq(), 1);
    }
}

TEST_F(ParticleTestHost, positron)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, TrackSlotId(0));
    particle = Initializer_t{ParticleId{3}, MevEnergy{1}};

    EXPECT_REAL_EQ(1, particle.energy().value());
    EXPECT_REAL_EQ(0.5109989461, particle.mass().value());
    EXPECT_REAL_EQ(1., particle.charge().value());
    EXPECT_REAL_EQ(0.0, particle.decay_constant());
    EXPECT_TRUE(particle.is_antiparticle());
    EXPECT_TRUE(particle.is_stable());
    EXPECT_FALSE(particle.is_heavy());
}

TEST_F(ParticleTestHost, gamma)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, TrackSlotId(0));
    particle = Initializer_t{ParticleId{1}, MevEnergy{1.234}};

    EXPECT_REAL_EQ(0, particle.mass().value());
    EXPECT_REAL_EQ(1.234, particle.energy().value());
    EXPECT_FALSE(particle.is_antiparticle());
    EXPECT_TRUE(particle.is_stable());
    EXPECT_REAL_EQ(1.234, particle.total_energy().value());
    EXPECT_EQ(1, particle.beta_sq());
    EXPECT_EQ(1, particle.speed().value());
    EXPECT_REAL_EQ(1.234, particle.momentum().value());
    EXPECT_FALSE(particle.is_heavy());
}

TEST_F(ParticleTestHost, neutron)
{
    ParticleTrackView particle(
        particle_params->host_ref(), state_ref, TrackSlotId(0));
    particle = Initializer_t{ParticleId{2}, MevEnergy{20}};

    EXPECT_REAL_EQ(20, particle.energy().value());
    EXPECT_REAL_EQ(1.0 / 879.4 * (1 / units::second),
                   particle.decay_constant());
    EXPECT_FALSE(particle.is_antiparticle());
    EXPECT_FALSE(particle.is_stable());
    EXPECT_TRUE(particle.is_heavy());
}

TEST_F(ParticleTestHost, speed)
{
    {
        SCOPED_TRACE("Low energy neutron");
        ParticleTrackView particle(
            particle_params->host_ref(), state_ref, TrackSlotId(0));
        particle = Initializer_t{ParticleId{2}, MevEnergy{1e-3}};

        EXPECT_SOFT_EQ(0.0014589860536436555,
                       value_as<units::LightSpeed>(particle.speed()));

        particle.energy(MevEnergy(1e-5));

        EXPECT_SOFT_EQ(0.00014589872066202113,
                       value_as<units::LightSpeed>(particle.speed()));
    }
    {
        SCOPED_TRACE("Low energy electron");
        ParticleTrackView particle(
            particle_params->host_ref(), state_ref, TrackSlotId(0));
        particle = Initializer_t{ParticleId{0}, MevEnergy{1e-5}};

        EXPECT_SOFT_EQ(0.0062560271021212888,
                       value_as<units::LightSpeed>(particle.speed()));

        particle.energy(MevEnergy(1e-18));

        EXPECT_GT(value_as<units::LightSpeed>(particle.speed()), 0);
    }
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
    real_type const expected_props[] = {0.5,
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
}  // namespace test
}  // namespace celeritas
