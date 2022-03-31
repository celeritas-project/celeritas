//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractionLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "sim/CoreTrackData.hh"
#include "sim/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to launch a "Track"-dependent function from core and
 * model data.
 *
 * This class should be used by generated interactor functions.
 */
template<class D, class F>
struct InteractionLauncherImpl
{
    //!@{
    //! Type aliases
    using CoreRefNative = CoreRef<MemSpace::native>;
    //!@}

    //// DATA ////

    CoreRefNative const& core_data;
    D const&         model_data;
    F                call_with_track;

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
InteractionLauncherImpl<D, F>::operator()(ThreadId thread) const
{
    CELER_ASSERT(thread < this->core_data.states.size());
    const celeritas::CoreTrackView track(
        this->core_data.params, this->core_data.states, thread);

    // TODO: will be replaced by action ID
    if (track.model_id() != model_data.ids.model)
        return;

    Interaction result  = this->call_with_track(model_data, track);
    track.interaction() = result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
