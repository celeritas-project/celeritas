//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Collection.test.cu
//---------------------------------------------------------------------------//
#include "Collection.test.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void col_cuda_test_kernel(DeviceCRef<MockParamsData> const params,
                                     DeviceRef<MockStateData> const states,
                                     Span<double> const results)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    // Initialize local matid states
    states.matid[tid] = MockMaterialId(tid.get() % params.materials.size());

    // Construct track view
    MockTrackView mock(params, states, tid);

    // Access some values
    MockMaterialId matid = mock.matid();
    CELER_ASSERT(matid);
    double nd = mock.number_density();
    CELER_ASSERT(nd >= 0);

    double result = 0;
    auto elements = mock.elements();
    if (!elements.empty())
    {
        MockElement const& el = elements[(tid.get() / 2) % elements.size()];
        result = matid.get() + nd * el.atomic_mass / el.atomic_number;
    }

    // Do a simple test of item range
    ItemRange<int> pr;
    pr = ItemRange<int>(ItemId<int>(123), ItemId<int>(456));
    if (pr.size() != 333)
    {
        // Failure
        result = -1;
    }

    results[tid.get()] = result;
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void col_cuda_test(CTestInput input)
{
    CELER_EXPECT(input.states.size() > 0);
    CELER_LAUNCH_KERNEL(col_cuda_test,
                        device().default_block_size(),
                        input.states.size(),
                        input.params,
                        input.states,
                        input.result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
