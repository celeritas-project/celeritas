//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.cc
//---------------------------------------------------------------------------//
#include "base/Pool.hh"

#include "celeritas_test.hh"
#include "Pool.test.hh"

using celeritas::PoolItem;
using celeritas::Pool;
using celeritas::Ownership;
using celeritas::MemSpace;

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PoolTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        host_pools.max_element_components = 3;
    }

    MockParamsPools<Ownership::value, MemSpace::host> host_pools;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PoolTest, all)
{
    // PTestInput input;
    // input.num_threads = 0;
    // auto result = p_test(input);
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}
