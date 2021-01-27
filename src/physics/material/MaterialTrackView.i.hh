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
    : params_(params), states_(states), tid_(id)
{
    CELER_EXPECT(id < states.state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION MaterialTrackView&
MaterialTrackView::operator=(const Initializer_t& other)
{
    CELER_EXPECT(other.def_id < params_.materials.size());
    this->state() = other;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Current material identifier.
 */
CELER_FORCEINLINE_FUNCTION MaterialId MaterialTrackView::def_id() const
{
    return this->state().def_id;
}

//---------------------------------------------------------------------------//
/*!
 * Get material properties for the current material.
 */
CELER_FORCEINLINE_FUNCTION MaterialView MaterialTrackView::material_view() const
{
    return MaterialView(params_, this->def_id());
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space with at least one real per element component.
 */
CELER_FUNCTION Span<real_type> MaterialTrackView::element_scratch()
{
    auto offset = tid_.get() * params_.max_element_components;
    CELER_ENSURE(offset + params_.max_element_components
                 <= states_.element_scratch.size());
    return {states_.element_scratch.data() + offset,
            params_.max_element_components};
}

//---------------------------------------------------------------------------//
/*!
 * Access the thread-local state.
 */
CELER_FORCEINLINE_FUNCTION MaterialTrackState& MaterialTrackView::state() const
{
    return states_.state[tid_.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
