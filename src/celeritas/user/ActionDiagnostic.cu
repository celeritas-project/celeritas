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
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/ActionDiagnosticImpl.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
tally_action_kernel(CRefPtr<CoreParamsData, MemSpace::device> const params,
                    RefPtr<CoreStateData, MemSpace::device> const state,
                    DeviceCRef<ParticleTallyParamsData> ad_params,
                    DeviceRef<ParticleTallyStateData> ad_state)
{
    auto execute = make_active_track_executor(
        *params, *state, detail::tally_action, ad_params, ad_state);
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void ActionDiagnostic::execute(CoreParams const& params,
                               CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(
        tally_action,
        celeritas::device().default_block_size(),
        state.size(),
        celeritas::device().stream(state.stream_id()).get(),
        params.ptr<MemSpace::native>(),
        state.ptr(),
        store_.params<MemSpace::device>(),
        store_.state<MemSpace::device>(state.stream_id(), this->state_size()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
