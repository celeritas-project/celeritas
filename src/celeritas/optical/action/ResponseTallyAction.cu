//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ResponseTallyAction.cu
//---------------------------------------------------------------------------//
#include "ResponseTallyAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.device.hh"
#include "TrackSlotExecutor.hh"

#include "detail/ResponseTallyExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void ResponseTallyAction::step(CoreParams const& params,
                               CoreStateDevice& state) const
{
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::ResponseTallyExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(), state.size())}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the start action with device data.
 */
void ResponseTallyStartAction::step(CoreParams const& params,
                                    CoreStateDevice& state) const
{
    auto& store = tally_->store();
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::ResponseTallyStartExecutor{
            store.params<MemSpace::native>(),
            store.state<MemSpace::native>(state.stream_id(), state.size())}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
