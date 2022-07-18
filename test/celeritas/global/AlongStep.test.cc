//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStep.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/alongstep/AlongStep.hh"

#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "../SimpleTestBase.hh"
#include "AlongStepTestBase.hh"
#include "celeritas_cmake_strings.h"
#include "celeritas_test.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class KnAlongStepTest : public celeritas_test::SimpleTestBase,
                        public celeritas_test::AlongStepTestBase
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(KnAlongStepTest, all)
{
    Input inp;
    inp.particle_id = this->particle()->find(pdg::gamma());
    inp.energy      = MevEnergy{1};
    auto result     = this->run(inp);
    result.print_expected();
}
