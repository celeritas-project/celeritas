//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include <celeritas/global/ActionLauncher.device.hh>
#include <celeritas/global/CoreParams.hh>
#include <celeritas/global/CoreState.hh>
#include <celeritas/global/TrackExecutor.hh>
#include <corecel/data/Filler.t.hh>

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
    auto& step_state = state.aux_data<MemSpace::native>(aux_id_);

    // Accumulate counters
    CoreStateCounters const& counters = state.counters();
    step_state.host_data.steps += counters.num_active;
    step_state.host_data.generated += counters.num_generated;
    step_state.host_data.secondaries += counters.num_secondaries;

    // Create a functor that gathers data from a single track slot
    auto execute
        = make_active_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     StepDiagnosticExecutor{step_state});

    // Gather kernel stats and run on all track slots
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
// Explicitly instantiate filler kernel from inside the .cu file
template class celeritas::Filler<NativeStepStatistics, MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
