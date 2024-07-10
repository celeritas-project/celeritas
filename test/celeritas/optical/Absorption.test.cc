//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Absorption.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/AbsorptionInteractor.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class AbsorptionInteractorTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(AbsorptionInteractorTest, basic)
{
    // A simple regression test to make sure the interaction is absorbed

    AbsorptionInteractor interact;
    OpticalInteraction result = interact();

    // Do a few checks to make sure there's no state
    for ([[maybe_unused]] int i : range(10))
    {
        EXPECT_EQ(OpticalInteraction::Action::absorbed, result.action);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
