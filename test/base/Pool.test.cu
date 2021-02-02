//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.cu
//---------------------------------------------------------------------------//
#define POOL_HOST_HEADER 0
#include "Pool.test.hh"

#include "base/KernelParamCalculator.cuda.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void p_test_kernel(
    const MockParamsPools<Ownership::const_reference, MemSpace::device> params,
    const MockStatePools<Ownership::reference, MemSpace::device>        states,
    const celeritas::Span<double> results)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize local matid states
    states.matid[tid.get()] = tid.get() % params.materials.size();

    // Construct track view
    MockTrackView mock(params, states, tid);

    // Access some values
    int matid = mock.matid();
    CELER_ASSERT(matid >= 0);
    double nd = mock.number_density();
    CELER_ASSERT(nd >= 0);

    double result   = 0;
    auto   elements = mock.elements();
    if (!elements.empty())
    {
        const MockElement& el = elements[tid % elements.size()];
        result                = matid + nd * el.atomic_mass / el.atomic_number
    }

    results[tid.get()] = result;
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
PTestOutput p_test(PTestInput input)
{
    PTestOutput output;
    output.result.resize(input.states.size());

    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.states.size());
    p_test_kernel<<<params.grid_size, params.block_size>>>(
        input.params, input.states, output.result.device_pointers());

    return output;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
