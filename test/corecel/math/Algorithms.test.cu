//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.test.cu
//---------------------------------------------------------------------------//
#include "Algorithms.test.hh"

#include "corecel/DeviceRuntimeApi.hh"

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

__global__ void alg_test_kernel(AlgorithmTestData data)
{
    auto tid = KernelParamCalculator::thread_id();

    auto const& inp = data.input;
    auto const& out = data.output;

    if (tid.get() < inp.pi_frac.size())
    {
        sincospi(inp.pi_frac[tid], &out.sinpi[tid], &out.cospi[tid]);
    }
    if (tid.get() < inp.a.size())
    {
        out.fastpow[tid] = fastpow(inp.a[tid], inp.b[tid]);
        out.hypot[tid] = hypot(inp.a[tid], inp.b[tid]);
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device
void alg_test(AlgorithmTestData data)
{
    CELER_LAUNCH_KERNEL(alg_test, data.num_threads, 0, data);
    CELER_DEVICE_API_CALL(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
