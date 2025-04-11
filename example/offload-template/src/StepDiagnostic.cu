//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include <celeritas/global/ActionLauncher.device.hh>
#include <celeritas/global/CoreParams.hh>
#include <celeritas/global/CoreState.hh>
#include <celeritas/global/TrackExecutor.hh>
#include <corecel/data/Filler.device.t.hh>

#include "StepDiagnosticData.hh"
#include "StepDiagnosticExecutor.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
// Launch a kernel from inside the .cu file
void StepDiagnostic::step(CoreParams const& params, CoreStateDevice& state) const
{
    auto const& step_params = this->ref<MemSpace::native>();
    auto& step_state = state.aux_data<StepStateData>(aux_id_);

    // Accumulate counters
    this->accum_counters(state.counters(), step_state.host_data);

    // Create a functor that gathers data from a single track slot
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        StepDiagnosticExecutor{step_params, step_state});

    // Gather kernel stats and run on all track slots
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace example

//---------------------------------------------------------------------------//
// Explicitly instantiate filler kernel from inside the .cu file
template class Filler<example::NativeStepStatistics, MemSpace::device>;

}  // namespace celeritas
