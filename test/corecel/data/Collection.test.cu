//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
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
                        input.states.size(),
                        0,
                        input.params,
                        input.states,
                        input.result);
}

//! Test that we can copy inside .cu code
template<Ownership W, MemSpace M>
MockStateData<Ownership::value, MemSpace::device>
copy_to_device_test(MockStateData<W, M>& inp)
{
    MockStateData<Ownership::value, MemSpace::device> result;
    result = inp;
    return result;
}

//! Test that we can make a device reference inside .cu code
MockStateData<Ownership::reference, MemSpace::device> reference_device_test(
    MockStateData<Ownership::value, MemSpace::device>& device_value)
{
    MockStateData<Ownership::reference, MemSpace::device> result;
    result = device_value;
    return result;
}

template MockStateData<Ownership::value, MemSpace::device>
copy_to_device_test<Ownership::value, MemSpace::host>(
    MockStateData<Ownership::value, MemSpace::host>&);
template MockStateData<Ownership::value, MemSpace::device>
copy_to_device_test<Ownership::reference, MemSpace::host>(
    MockStateData<Ownership::reference, MemSpace::host>&);
template MockStateData<Ownership::value, MemSpace::device>
copy_to_device_test<Ownership::const_reference, MemSpace::host>(
    MockStateData<Ownership::const_reference, MemSpace::host>&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
