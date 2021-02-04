//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.cc
//---------------------------------------------------------------------------//
#include "base/Pool.hh"
#include "base/PoolBuilder.hh"

#include "celeritas_test.hh"
#include "Pool.test.hh"
#include "base/DeviceVector.hh"
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

        auto el_builder  = make_pool_builder(host_pools.elements);
        auto mat_builder = make_pool_builder(host_pools.materials);
        el_builder.reserve(5);
        mat_builder.reserve(2);

        //// Construct materials and elements ////
        {
            MockMaterial m;
            m.number_density             = 2.0;
            const MockElement elements[] = {
                {1, 1.1},
                {3, 5.0},
                {6, 12.0},
            };
            m.elements = el_builder.insert_back(std::begin(elements),
                                                std::end(elements));
            EXPECT_EQ(3, m.elements.size());
            mat_builder.push_back(m);
        }
        {
            MockMaterial m;
            m.number_density = 20.0;
            m.elements       = el_builder.insert_back({{10, 20.0}});
            mat_builder.push_back(m);
        }
        {
            MockMaterial m;
            m.number_density = 0.0;
            mat_builder.push_back(m);
        }
        EXPECT_EQ(3, mat_builder.size());
        EXPECT_EQ(3, host_pools.materials.size());
        EXPECT_EQ(4, host_pools.elements.size());

        //// Create host reference ////

        mock_params.host_ref = mock_params.host;

        //// Copy to device ////

        if (celeritas::is_device_enabled())
        {
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

    make_pool_builder(host_state.matid).resize(1);
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
    make_pool_builder(device_states.matid).resize(1024);

    celeritas::DeviceVector<double> device_result(device_states.size());

    PTestInput kernel_input;
    kernel_input.params = mock_params.device_ref;
    kernel_input.states = device_states;
    kernel_input.result = device_result.device_pointers();

#if CELERITAS_USE_CUDA
    p_test(kernel_input);
#endif
    std::vector<double> result(device_result.size());
    device_result.copy_to_host(celeritas::make_span(result));

    // For brevity, only check the first 6 values (they repeat after that)
    result.resize(6);
    const double expected_result[] = {2.2, 41, 0, 3.333333333333, 41, 0};
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}
