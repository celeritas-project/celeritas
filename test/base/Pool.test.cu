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
    if (tid.get() >= states.size())
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
        const MockElement& el = elements[(tid.get() / 2) % elements.size()];
        result                = matid + nd * el.atomic_mass / el.atomic_number;
    }

    // Do a stupid test of pool range
    PoolRange<int> pr;
    pr = PoolRange<int>(123, 456);
    if (pr.size() != 333)
    {
        // Failure
        result = -1;
    }

    results[tid.get()] = result;
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void p_test(PTestInput input)
{
    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.states.size());
    p_test_kernel<<<params.grid_size, params.block_size>>>(
        input.params, input.states, input.result);

    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
