//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "detail/StepDiagnosticImpl.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
tally_steps_kernel(DeviceCRef<CoreParamsData> const params,
                   DeviceRef<CoreStateData> const state,
                   DeviceCRef<ParticleTallyParamsData> sd_params,
                   DeviceRef<ParticleTallyStateData> sd_state)
{
    auto launch = make_active_track_launcher(
        params, state, detail::tally_steps, sd_params, sd_state);
    launch(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void StepDiagnostic::execute(CoreParams const& params,
                             CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(
        tally_steps,
        celeritas::device().default_block_size(),
        state.size(),
        params.ref<MemSpace::native>(),
        state.ref(),
        store_.params<MemSpace::device>(),
        store_.state<MemSpace::device>(state.stream_id(), this->state_size()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
