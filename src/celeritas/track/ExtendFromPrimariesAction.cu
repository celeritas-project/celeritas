//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/ProcessPrimariesExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to create track initializers from primary particles.
 */
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const& params,
    CoreStateDevice& state,
    PrimaryStateData<MemSpace::device> const& pstate) const
{
    auto primaries = pstate.primaries();
    auto counters = state.sync_get_counters();
    detail::ProcessPrimariesExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), counters, primaries};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
    if (!primaries.empty())
    {
        launch_kernel(primaries.size(), state.stream_id(), execute_thread);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct primaries.
 */
void ExtendFromPrimariesAction::step_impl(CoreParams const& params,
                                          CoreStateDevice& state) const
{
    auto& primaries
        = get<PrimaryStateData<MemSpace::device>>(state.aux(), aux_id_);
    auto counters = state.sync_get_counters();

    // Create track initializers from primaries
    counters.num_initializers += primaries.count;
    state.sync_put_counters(counters);
    this->process_primaries(params, state, primaries);

    // Mark that the primaries have been processed
    counters.num_generated += primaries.count;
    counters.num_pending = 0;
    primaries.count = 0;
    state.sync_put_counters(counters);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
