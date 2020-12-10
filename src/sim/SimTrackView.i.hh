//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimTrackView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and local data.
 */
CELER_FUNCTION
SimTrackView::SimTrackView(const SimStatePointers& states, ThreadId id)
    : state_(states.vars[id.get()])
{
    REQUIRE(id < states.vars.size());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the particle.
 */
CELER_FUNCTION SimTrackView& SimTrackView::operator=(const Initializer_t& other)
{
    state_ = other;
    return *this;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
