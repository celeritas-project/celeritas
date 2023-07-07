//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/NoMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreTrackView;
struct StepLimit;

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

    //! Get the maximum required safety radius before step limiting
    CELER_FUNCTION real_type safety_pre(CoreTrackView const&) const
    {
        return 0;
    }

    //! No updates needed to the physical and geometric step lengths
    CELER_FUNCTION void limit_step(CoreTrackView const&, StepLimit*) const {}

    //! MSC is never applied
    CELER_FUNCTION void apply_step(CoreTrackView const&, StepLimit*) const {}

    //! Get the maximum required safety radius after step limiting
    CELER_FUNCTION real_type safety_post(CoreTrackView const&) const
    {
        return 0;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
