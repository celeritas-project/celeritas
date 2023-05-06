//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnostic.cu
//---------------------------------------------------------------------------//
#include "ActionDiagnostic.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/TrackLauncher.hh"
+ #include "celeritas/global/CoreParams.hh"

#include "detail/ActionDiagnosticImpl.hh"

    namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
tally_action_kernel(DeviceCRef<CoreParamsData> const params,
                    DeviceRef<CoreStateData> const state,
                    DeviceCRef<ParticleTallyParamsData> ad_params,
                    DeviceRef<ParticleTallyStateData> ad_state)
{
    auto launch = make_active_track_launcher(
        params, state, detail::tally_action, ad_params, ad_state);
    launch(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void ActionDiagnostic::execute(CoreParams const& params, StateDeviceRef& state)
    const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    if (!store_)
    {
        this->build_stream_store();
    }
    CELER_LAUNCH_KERNEL(
        tally_action,
        celeritas::device().default_block_size(),
        state.size(),
        params.ref<MemSpace::native>(),
        state,
        store_.params<MemSpace::device>(),
        store_.state<MemSpace::device>(state.stream_id, this->state_size()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
