//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepNeutral.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/geo/GeoTrackView.hh"

#include "../AlongStep.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for along-step kernel when no MSC is in use.
 */
struct NoMsc
{
    //! MSC never applies to the current track
    CELER_FUNCTION bool
    is_applicable(CoreTrackView const&, real_type step) const
    {
        CELER_ASSERT(step > 0);
        return false;
    }

    //! No updates needed to the physical and geometric step lengths
    CELER_FUNCTION void limit_step(CoreTrackView const&, StepLimit*) const {}

    //! MSC is never applied
    CELER_FUNCTION void apply_step(CoreTrackView const&, StepLimit*) const {}
};

//---------------------------------------------------------------------------//
/*!
 * Create a propagator for neutral particles or no fields.
 */
struct LinearPropagatorFactory
{
    CELER_FUNCTION decltype(auto)
    operator()(ParticleTrackView const&, GeoTrackView* geo) const
    {
        return LinearPropagator{geo};
    };
};

//---------------------------------------------------------------------------//
/*!
 * Class that returns zero energy loss.
 */
struct NoELoss
{
    //! This calculator never returns energy loss
    CELER_CONSTEXPR_FUNCTION bool is_applicable(CoreTrackView const&)
    {
        return false;
    }

    //! No energy loss
    CELER_FUNCTION auto calc_eloss(CoreTrackView const&, real_type, bool) const
        -> decltype(auto)
    {
        return zero_quantity();
    }

    //! No slowing down
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return false; }
};

//---------------------------------------------------------------------------//
struct AlongStepNeutralExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Implementation of the "along step" action for neutral particles.
 *
 * This is mostly a demonstration use case and *should not* be used as part of
 * a complete EM shower simulation because it currently applies to *all*
 * particles as opposed to just neutral ones.
 *
 * This will be called by \c make_active_track_executor inside a generated
 * kernel:
 * \code
 * auto execute = make_active_track_executor(params, state,
 * along_step_neutral); \endcode
 */
CELER_FUNCTION void
AlongStepNeutralExecutor::operator()(CoreTrackView const& track)
{
    return along_step(track, NoMsc{}, LinearPropagatorFactory{}, NoELoss{});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
