//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cc
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include "celeritas_test.hh"   // for CELER_EXPECT and Test

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DormandPrinceStepperTest : public Test
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DormandPrinceStepperTest, gpu)
{
    dormand_prince_cuda_test();
    // CELER_LOG(info) << "Time taken by function: some nanoseconds";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
