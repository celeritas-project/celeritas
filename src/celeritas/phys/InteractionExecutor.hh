//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreTrackDataFwd.hh"

#include "detail/InteractionExecutorImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a function-like class to execute a "Track"-dependent function.
 *
 * This function should be used exclusively by generated interaction functions.
 * The author of a new model needs to define (in a header file) an inline
 * function that accepts a CoreTrackView and a const reference to "native"
 * model data, and returns an Interaction. It is a given that the interaction
 * *does* apply to the given track.
 *
 * \deprecated Use \c InteractionApplier with \c make_action_track_executor .
 */
template<class D, class F>
CELER_FUNCTION detail::InteractionExecutorImpl<D, F>
make_interaction_executor(CRefPtr<CoreParamsData, MemSpace::native> const& params,
                          RefPtr<CoreStateData, MemSpace::native> const& state,
                          F&& call_with_track,
                          D const& model_data)
{
    return {
        params, state, ::celeritas::forward<F>(call_with_track), model_data};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
