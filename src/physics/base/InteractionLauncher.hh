//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractionLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "sim/CoreTrackData.hh"

#include "detail/InteractionLauncherImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a function-like class to launch a "Track"-dependent function.
 *
 * This function should be used exclusively by generated interaction functions.
 * The author of a new model needs to define (in a header file) an inline
 * function that accepts a CoreTrackView and a const reference to "native"
 * model data, and returns an Interaction. It is a given that the interaction
 * *does* apply to the given track.
 *
 * \code
inline CELER_FUNCTION Interaction foo_interact(
    FooModelData<Ownership::const_reference, MemSpace::native> const& data,
    celeritas::CoreTrackView const& track)
{
    // ...
}
   \endcode
 *
 * \note The model data *must* have a member data `ModelId model_id;` for
 * filtering the tracks. We could improve this interface later.
 */
template<class D, class F>
CELER_FUNCTION detail::InteractionLauncherImpl<D, F> make_interaction_launcher(
    CoreParamsData<Ownership::const_reference, MemSpace::native> const& params,
    CoreStateData<Ownership::reference, MemSpace::native> const&        state,
    D const& model_data,
    F&&      call_with_track)
{
    return {
        params, state, model_data, ::celeritas::forward<F>(call_with_track)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
