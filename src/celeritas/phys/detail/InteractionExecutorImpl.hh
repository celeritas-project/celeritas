//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/InteractionExecutorImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../InteractionApplier.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to execute a "Track"-dependent function from core and
 * model data.
 *
 * This class should be used by generated interactor functions.
 */
template<class D, class F>
struct InteractionExecutorImpl
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    F call_with_track;
    D const& model_data;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply the interaction to the track with the given thread ID.
 */
template<class D, class F>
CELER_FUNCTION void
InteractionExecutorImpl<D, F>::operator()(ThreadId thread) const
{
    CELER_EXPECT(thread < state->size());
    celeritas::CoreTrackView const track(*params, *state, thread);

    auto sim = track.make_sim_view();
    if (sim.step_limit().action != model_data.ids.action)
        return;

    // Wrap the "call_with_track" function
    InteractionApplier apply_impl{[this](celeritas::CoreTrackView const& track) {
        return call_with_track(model_data, track);
    }};
    return apply_impl(track);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
