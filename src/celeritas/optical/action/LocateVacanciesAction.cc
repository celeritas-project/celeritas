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
void LocateVacanciesAction::step(CoreParams const&, CoreStateDevice& state) const
{
    return this->step_impl(state);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize optical track states.
 */
template<MemSpace M>
void LocateVacanciesAction::step_impl(CoreState<M>& state) const
{
    auto counters = state.sync_get_counters();

    // Compact the IDs of the inactive tracks, getting the sorted indices of
    // the empty slots
    counters.num_vacancies = detail::copy_if_vacant(
        state.ref().sim.status, state.ref().init.vacancies, state.stream_id());

    counters.num_alive = state.size() - counters.num_vacancies;
    state.sync_put_counters(counters);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
