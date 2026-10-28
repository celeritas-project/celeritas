//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/LocateVacanciesAction.cc
//---------------------------------------------------------------------------//
#include "LocateVacanciesAction.hh"

#include "celeritas/optical/CoreState.hh"
#include "celeritas/track/CounterAlgorithms.hh"

#include "detail/TrackInitAlgorithms.hh"

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
void LocateVacanciesAction::step(CoreParams const&, CoreStateHost& state) const
{
    return this->step_impl(state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void LocateVacanciesAction::step(CoreParams const&,
                                 CoreStateDevice& state) const
{
    return this->step_impl(state);
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
    return celeritas::update_alive(
        state.ref().init.counters, state.size(), state.stream_id());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
