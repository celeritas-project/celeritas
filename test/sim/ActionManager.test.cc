//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ActionManager.test.cc
//---------------------------------------------------------------------------//
#include "sim/ActionManager.hh"

#include "celeritas_test.hh"

using celeritas::ActionManager;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ActionManagerTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ActionManagerTest, host)
{
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}

// TEST_F(ActionManagerTest, TEST_IF_CELER_DEVICE(device))
// {
//     AMTestInput input;
//     am_test(input);
// }
