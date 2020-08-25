//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantImporter.test.cc
//---------------------------------------------------------------------------//
#include "io/GeantImporter.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::GeantImporter;
using celeritas::GeantParticle;
using celeritas::ParticleDef;
using celeritas::ParticleDefId;
using celeritas::ParticleParams;
using celeritas::PDGNumber;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeantImporterTest : public celeritas::Test
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
TEST_F(GeantImporterTest, import_particles)
{
    GeantImporter import(root_filename_.c_str());
    auto          data = import();

    EXPECT_EQ(19, data.particle_params->size());

    EXPECT_GE(data.particle_params->find(PDGNumber(11)).get(), 0);
    ParticleDefId electron_id = data.particle_params->find(PDGNumber(11));
    ParticleDef   electron    = data.particle_params->get(electron_id);

    EXPECT_SOFT_EQ(0.510998910, electron.mass);
    EXPECT_EQ(-1, electron.charge);
    EXPECT_EQ(0, electron.decay_constant);

    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (const auto& md : data.particle_params->md())
    {
        loaded_names.push_back(md.name);
        loaded_pdgs.push_back(md.pdg_code.get());
    }

    // clang-format off
    const std::string expected_loaded_names[] = {"gamma", "e+", "e-", "mu+",
        "mu-", "pi-", "pi+", "kaon-", "kaon+", "anti_proton", "proton",
        "anti_deuteron", "deuteron", "anti_He3", "He3", "anti_triton",
        "triton", "anti_alpha", "alpha"};
    const int expected_loaded_pdgs[] = {22, -11, 11, -13, 13, -211, 211, -321,
        321, -2212, 2212, -1000010020, 1000010020, -1000020030, 1000020030,
        -1000010030, 1000010030, -1000020040, 1000020040};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}
