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
#include "comm/Device.hh"

using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::Pool;
using celeritas::Span;

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PoolTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        MockParamsPools<Ownership::value, MemSpace::host>& host_pools
            = mock_params.host;
        host_pools.max_element_components = 3;
        host_pools.elements.reserve(5);
        host_pools.materials.reserve(2);

        // Assign materials
        auto               mat_range = host_pools.materials.allocate(3);
        Span<MockMaterial> mats      = host_pools.materials[mat_range];
        ASSERT_EQ(3, mats.size());

        {
            mats[0].number_density     = 2.0;
            mats[0].elements           = host_pools.elements.allocate(3);
            Span<MockElement> elements = host_pools.elements[mats[0].elements];
            ASSERT_EQ(3, elements.size());
            elements[0] = {1, 1.1};
            elements[1] = {3, 5.0};
            elements[2] = {6, 12.0};
        }

        {
            mats[1].number_density     = 20.0;
            mats[1].elements           = host_pools.elements.allocate(1);
            Span<MockElement> elements = host_pools.elements[mats[1].elements];
            ASSERT_EQ(1, elements.size());
            elements[0] = {10, 20.0};
        }

        {
            mats[2].number_density = 0.0;
        }

        mock_params.host_ref = mock_params.host;
        if (celeritas::is_device_enabled())
        {
            // Copy to GPU
            mock_params.device     = mock_params.host;
            mock_params.device_ref = mock_params.device;
        }
    }

    CELER_POOL_STRUCT(MockParamsPools, const_reference) mock_params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PoolTest, host)
{
    MockStatePools<Ownership::value, MemSpace::host>     host_state;
    MockStatePools<Ownership::reference, MemSpace::host> host_state_ref;

    host_state.matid.allocate(1);
    host_state_ref = host_state;

    // Assign
    host_state_ref.matid[0] = 1;

    // Create view
    MockTrackView mock(
        mock_params.host_ref, host_state_ref, celeritas::ThreadId{0});
    EXPECT_EQ(1, mock.matid());
}

TEST_F(PoolTest, device)
{
    if (!celeritas::is_device_enabled())
    {
        SKIP("GPU capability is disabled");
    }

    // Construct with 1024 states
    MockStatePools<Ownership::value, MemSpace::device> device_states;
    device_states.matid.resize(1024);

    PTestInput kernel_input;
    kernel_input.params = mock_params.device_ref;
    kernel_input.states = device_states;

    PTestOutput output;
#if CELERITAS_USE_CUDA
    output = p_test(kernel_input);
#endif
    std::vector<double> result(output.result.size());
    output.result.copy_to_host(celeritas::make_span(result));

    // For brevity, only check the first 32 values
    result.resize(32);
    PRINT_EXPECTED(result);
}
