//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/RngReseed.cu
//---------------------------------------------------------------------------//
#include "RngReseed.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device using the Geant4 Event ID.
 */
__global__ void reseed_rng_kernel(DeviceCRef<RngParamsData> const params,
                                  DeviceRef<RngStateData> const state,
                                  size_type event_id)
{
    auto tid = TrackSlotId{
        celeritas::KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < state.size())
    {
        TrackSlotId tsid{tid.unchecked_get()};
        RngEngine::Initializer_t init;
#if CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW
        init.seed = params.seed[0];
#else
        init.seed = params.seed;
#endif
        init.subsequence = event_id * state.size() + tsid.get();
        RngEngine rng(params, state, tsid);
        rng = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device using the Geant4 Event ID.
 *
 * Each thread's state is initialized using same seed and skipped ahead a
 * different number of subsequences so the sequences on different threads will
 * not have statistically correlated values.
 */
void reseed_rng(DeviceCRef<RngParamsData> const& params,
                DeviceRef<RngStateData> const& state,
                size_type event_id)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_LAUNCH_KERNEL(reseed_rng, state.size(), 0, params, state, event_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
