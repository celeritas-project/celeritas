//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to launch a "Track"-dependent function from core and
 * model data.
 *
 * \tparam M MSC params
 * \tparam P Propagator params (e.g. field definitions)
 * \tparam E Energy loss params
 * \tparam F Launcher function
 */
template<class M, class P, class E, class F>
struct AlongStepLauncherImpl
{
    //!@{
    //! \name Type aliases
    using CoreRefNative = CoreRef<MemSpace::native>;
    //!@}

    //// DATA ////

    CoreRefNative const& core_data;
    M msc_data;
    P propagator_data;
    E eloss_data;
    F call_with_track;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply the interaction to the track with the given thread ID.
 */
template<class M, class P, class E, class F>
CELER_FUNCTION void
AlongStepLauncherImpl<M, P, E, F>::operator()(ThreadId thread) const
{
    CELER_ASSERT(thread < this->core_data.states.size());
    const celeritas::CoreTrackView track(
        this->core_data.params, this->core_data.states, thread);

    {
        auto sim = track.make_sim_view();
        if (sim.status() == TrackStatus::inactive)
        {
            // Track slot is empty
            CELER_ASSERT(!sim.step_limit());
            return;
        }
        CELER_ASSERT(sim.status() != TrackStatus::killed);
    }

    this->call_with_track(msc_data, propagator_data, eloss_data, track);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
