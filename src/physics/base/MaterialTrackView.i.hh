//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialTrackView.i.hh
//---------------------------------------------------------------------------//

#include <cmath>

#include "base/Assert.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
MaterialTrackView::MaterialTrackView(const MaterialParamsPointers& params,
                                     const MaterialStatePointers&  states,
                                     ThreadId                      id)
    : params_(params), state_(states.state[id.get()])
{
    REQUIRE(id < states.state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION MaterialTrackView&
MaterialTrackView::operator=(const Initializer_t& other)
{
    REQUIRE(other.def_id < params_.materials.size());
    state_ = other;
    return *this;
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Get static material defs for the current state.
 */
CELER_FUNCTION const MaterialDef& MaterialTrackView::material_def() const
{
    REQUIRE(state_.def_id < params_.materials.size());
    return params_.materials[state_.def_id.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
