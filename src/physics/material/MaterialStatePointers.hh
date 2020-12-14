//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialStatePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "MaterialDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Dynamic material state of a particle track.
 */
struct MaterialTrackState
{
    MaterialDefId def_id; //!< Current material being tracked
};

//---------------------------------------------------------------------------//
/*!
 * View to the dynamic states of multiple physical particles.
 *
 * The size of the view will be the size of the vector of tracks. Each particle
 * track state corresponds to the thread ID (\c ThreadId).
 *
 * The "element scratch space" is a 2D array of reals, indexed with
 * [track_id][el_component_id], where the fast-moving dimension has the
 * greatest number of element components of any material in the problem. This
 * can be used for the physics to calculate microscopic cross sections.
 *
 * \sa MaterialStateStore (owns the pointed-to data)
 * \sa MaterialTrackView (uses the pointed-to data in a kernel)
 */
struct MaterialStatePointers
{
    Span<MaterialTrackState> state;
    Span<real_type> element_scratch; // 2D array: [num states][max components]

    //! Check whether the view is assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
