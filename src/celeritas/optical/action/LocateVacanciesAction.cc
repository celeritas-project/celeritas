//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/LocateVacanciesAction.cc
//---------------------------------------------------------------------------//
#include "LocateVacanciesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackExecutor.hh"

#include "ActionLauncher.hh"

#include "detail/TrackInitAlgorithms.hh"
#include "detail/UpdateAliveExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
LocateVacanciesAction::LocateVacanciesAction(ActionId aid)
    : ConcreteAction(aid, "locate-vacancies", "locate vacant track states")
{
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void LocateVacanciesAction::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    this->step_impl(state);
    return this->update_alive(params, state, state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void LocateVacanciesAction::step(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    this->step_impl(state);
    return this->update_alive(params, state, state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Compact the IDs of the inactive slots to find the vacancies and update the
 * number of alive slots accordingly.
 */
template<MemSpace M>
void LocateVacanciesAction::step_impl(CoreState<M>& state) const
{
    // Compact the IDs of the inactive tracks, getting the sorted indices of
    // the empty slots
    detail::copy_if_vacant(
        state.ref().sim.status, state.ref().init, state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Update the number of alive slots as the empty slots have been compacted.
 */
void LocateVacanciesAction::update_alive(
    CoreParams const& params, CoreStateHost& state, size_type state_size) const
{
    auto execute_thread
        = make_single_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     detail::UpdateAliveExecutor{state_size});
    launch_action(1, execute_thread);
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void LocateVacanciesAction::update_alive(
    CoreParams const&, CoreStateDevice&, size_type) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
