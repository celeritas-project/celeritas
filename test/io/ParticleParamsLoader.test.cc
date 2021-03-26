//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParamsLoader.test.cc
//---------------------------------------------------------------------------//
#include "io/ParticleParamsLoader.hh"
#include "physics/base/PDGNumber.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ParticleParamsLoaderTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
    }
    std::string root_filename_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(ParticleParamsLoaderTest, load_particle_params)
{
    RootLoader           root_loader(this->root_filename_.c_str());
    ParticleParamsLoader part_params_loader(root_loader);

    const auto particles = part_params_loader();

    EXPECT_EQ(19, particles->size());

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
