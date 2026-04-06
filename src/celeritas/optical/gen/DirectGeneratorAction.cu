//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/DirectGeneratorAction.cu
//---------------------------------------------------------------------------//
#include "DirectGeneratorAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "detail/DirectGeneratorExecutor.hh"
#include "detail/UpdatePendingExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to initialize optical photons.
 */
void DirectGeneratorAction::generate(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<DirectGeneratorState<MemSpace::native>>(
        *state.aux(), this->aux_id());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::DirectGeneratorExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), aux_state.store.ref()};
    static ActionLauncher<decltype(execute)> const launch(*this);
    launch(num_gen, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to initialize optical photons.
 */
void DirectGeneratorAction::update_pending(CoreStateDevice& state,
                                           size_type num_pending) const
{
    // Update the number of pending optical photons
    detail::UpdatePendingExecutor execute{state.ptr(), num_pending};
    static KernelLauncher<decltype(execute)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
