//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStep.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/TestEm3Base.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "../SimpleTestBase.hh"
#include "AlongStepTestBase.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class KnAlongStepTest : public celeritas_test::SimpleTestBase,
                        public celeritas_test::AlongStepTestBase
{
  public:
};

#define Em3AlongStepTest TEST_IF_CELERITAS_GEANT(Em3AlongStepTest)
class Em3AlongStepTest : public celeritas_test::TestEm3Base,
                         public celeritas_test::AlongStepTestBase
{
  public:
    bool enable_msc() const override { return msc_; }
    bool enable_fluctuation() const override { return fluct_; }

    bool msc_{false};
    bool fluct_{true};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(KnAlongStepTest, basic)
{
    size_type num_tracks = 10;
    Input     inp;
    inp.particle_id = this->particle()->find(pdg::gamma());
    {
        inp.energy  = MevEnergy{1};
        auto result = this->run(inp, num_tracks);
        EXPECT_SOFT_EQ(0, result.eloss);
        EXPECT_SOFT_EQ(5, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(1.6678204759908e-10, result.time);
        EXPECT_SOFT_EQ(5, result.step);
    }
    {
        inp.energy  = MevEnergy{10};
        auto result = this->run(inp, num_tracks);
        EXPECT_SOFT_EQ(0, result.eloss);
        EXPECT_SOFT_EQ(5, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(1.6678204759908e-10, result.time);
        EXPECT_SOFT_EQ(5, result.step);
    }
}

TEST_F(Em3AlongStepTest, nofluct_nomsc)
{
    msc_   = false;
    fluct_ = false;

    size_type num_tracks = 128;
    Input     inp;
    inp.direction = {1, 0, 0};
    {
        SCOPED_TRACE("electron far from boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 0.25};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.46842934015656, result.eloss, 5e-4);
        EXPECT_SOFT_EQ(0.25, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(8.348974534499e-12, result.time);
        EXPECT_SOFT_EQ(0.25, result.step);
    }
    {
        SCOPED_TRACE("electron very near (1um) boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 1e-4};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.00018784530172589, result.eloss, 5e-4);
        EXPECT_SOFT_EQ(0.0001, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(3.3395898137996e-15, result.time);
        EXPECT_SOFT_EQ(0.0001, result.step);
    }
}

TEST_F(Em3AlongStepTest, msc_nofluct)
{
    msc_   = true;
    fluct_ = false;

    size_type num_tracks = 1024;
    Input     inp;
    inp.direction = {1, 0, 0};
    {
        SCOPED_TRACE("electron far from boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 0.25};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.47496066261651, result.eloss, 5e-4);
        EXPECT_SOFT_EQ(0.25, result.displacement);
        EXPECT_SOFT_NEAR(0.86687445437576, result.angle, 1e-3);
        EXPECT_SOFT_NEAR(8.4653845033461e-12, result.time, 1e-5);
        EXPECT_SOFT_NEAR(0.25348575649519, result.step, 1e-5);
    }
    {
        SCOPED_TRACE("low energy electron far from boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{1};
        inp.position    = {0.0 - 0.25};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.28579817262705, result.eloss, 5e-4);
        EXPECT_SOFT_NEAR(0.13028709259427, result.displacement, 5e-4);
        EXPECT_SOFT_NEAR(0.42060290539404, result.angle, 1e-3);
        EXPECT_SOFT_EQ(5.3240431819014e-12, result.time);
        EXPECT_SOFT_EQ(0.1502064087009, result.step);
    }
    {
        SCOPED_TRACE("electron very near (1um) boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 1e-4};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.00018784630366397, result.eloss, 1e-4);
        EXPECT_SOFT_EQ(0.0001, result.displacement);
        EXPECT_SOFT_NEAR(0.87210603989396, result.angle, 1e-3);
        EXPECT_SOFT_EQ(3.3396076266578e-15, result.time);
        EXPECT_SOFT_NEAR(0.00010000053338461, result.step, 1e-8);
    }
}

TEST_F(Em3AlongStepTest, fluct_nomsc)
{
    msc_   = false;
    fluct_ = true;

    size_type num_tracks = 4096;
    Input     inp;
    inp.direction = {1, 0, 0};
    {
        SCOPED_TRACE("electron far from boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 0.25};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.4682011732253, result.eloss, 2e-3);
        EXPECT_SOFT_EQ(0.25, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(8.3489745344989e-12, result.time);
        EXPECT_SOFT_EQ(0.25, result.step);
    }
    {
        SCOPED_TRACE("electron very near (1um) boundary");
        inp.particle_id = this->particle()->find(pdg::electron());
        inp.energy      = MevEnergy{10};
        inp.position    = {0.0 - 1e-4};
        auto result     = this->run(inp, num_tracks);
        EXPECT_SOFT_NEAR(0.00019264335626186, result.eloss, 0.1);
        EXPECT_SOFT_EQ(9.9999999999993e-05, result.displacement);
        EXPECT_SOFT_EQ(1, result.angle);
        EXPECT_SOFT_EQ(3.3395898137995e-15, result.time);
        EXPECT_SOFT_EQ(9.9999999999993e-05, result.step);
    }
}
