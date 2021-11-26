//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Data.hh"
#include "orange/surfaces/Surfaces.hh"
#include "detail/LogicEvaluator.hh"
#include "detail/SenseCalculator.hh"
#include "detail/Types.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Track a particle in a universe of well-connected volumes.
 *
 * The simple unit tracker is based on a set of non-overlapping volumes
 * comprised of surfaces. It is a faster but less "user-friendly" version of
 * the masked unit tracker because it requires all volumes to be exactly
 * defined by their connected surfaces. It does *not* check for overlaps.
 */
class SimpleUnitTracker
{
  public:
    //!@{
    //! Type aliases
    using ParamsRef
        = OrangeParamsData<Ownership::const_reference, MemSpace::native>;
    using Initialization = detail::Initialization;
    using LocalState     = detail::LocalState;
    //!@}

  public:
    // Construct with parameters (surfaces, cells)
    inline CELER_FUNCTION SimpleUnitTracker(const ParamsRef& params);

    // Find the local cell and possibly surface ID.
    inline CELER_FUNCTION Initialization initialize(LocalState state) const;

  private:
    const ParamsRef& params_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent parameter data.
 *
 * \todo When adding multiple universes, this will calculate range of VolumeIds
 * that belong to this unit. For now we assume all volumes and surfaces belong
 * to us.
 */
CELER_FUNCTION SimpleUnitTracker::SimpleUnitTracker(const ParamsRef& params)
    : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the local cell and possibly surface ID.
 *
 * This function is valid for initialization from a point, *and* for
 * initialization across a boundary.
 */
CELER_FUNCTION auto SimpleUnitTracker::initialize(LocalState state) const
    -> Initialization
{
    CELER_EXPECT(params_);

    detail::SenseCalculator calc_senses(
        Surfaces{params_.surfaces}, state.pos, state.temp_senses);

    for (VolumeId volid : range(VolumeId{params_.volumes.size()}))
    {
        if (state.surface && volid == state.volume)
        {
            // Cannot cross surface into the same cell
            continue;
        }

        VolumeView vol{params_.volumes, volid};

        // Calculate the local senses and face, and see if we're inside.
        auto logic_state
            = calc_senses(vol, detail::find_face(vol, state.surface));
        bool found = detail::LogicEvaluator(vol.logic())(logic_state.senses);
        if (!found)
        {
            // Try the next cell
            continue;
        }
        if (!state.surface && logic_state.face)
        {
            // Initialized on a boundary in this cell but wasn't known
            // to be crossing a surface. Fail safe by letting the multi-level
            // tracking geometry bump and try again.
            break;
        }

        // Found and not unexpectedly on a boundary!
        return {volid, get_surface(vol, logic_state.face)};
    }

    // Failed to find a valid volume containing the point
    return {};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
