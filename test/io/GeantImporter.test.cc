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

    EXPECT_EQ(20, data.particle_params->size());

    EXPECT_GE(data.particle_params->find(PDGNumber(11)).get(), 0);
    ParticleDefId electron_id = data.particle_params->find(PDGNumber(11));
    ParticleDef   electron    = data.particle_params->get(electron_id);

    EXPECT_SOFT_EQ(0.510998910, electron.mass);
    EXPECT_EQ(-1, electron.charge);
    EXPECT_EQ(0, electron.decay_constant);
}
