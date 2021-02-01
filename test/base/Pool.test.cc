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

using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::Pool;

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
        host_pools.elements.reserve(3);
        host_pools.materials.reserve(4);

        host_pools.elements.allocate(3);
    }

    MockParamsPools<Ownership::value, MemSpace::host>       host_pools;
    MockParamsPools<Ownership::value, MemSpace::device>     device_pools;
    MockParamsPools<Ownership::const_reference, MemSpace::host>   host_ptrs;
    MockParamsPools<Ownership::const_reference, MemSpace::device> device_ptrs;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PoolTest, host)
{
    // Create views
    host_ptrs = host_pools;
}

TEST_F(PoolTest, device)
{
    device_pools = host_pools;
}
