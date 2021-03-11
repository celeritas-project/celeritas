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
                                     ThreadId                      tid)
    : params_(params), states_(states), thread_(tid)
{
    CELER_EXPECT(tid < states.state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION MaterialTrackView&
MaterialTrackView::operator=(const Initializer_t& other)
{
    CELER_EXPECT(other.material_id < params_.materials.size());
    this->state() = other;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Current material identifier.
 */
CELER_FORCEINLINE_FUNCTION MaterialId MaterialTrackView::material_id() const
{
    return this->state().material_id;
}

//---------------------------------------------------------------------------//
/*!
 * Get material properties for the current material.
 */
CELER_FORCEINLINE_FUNCTION MaterialView MaterialTrackView::material_view() const
{
    return MaterialView(params_, this->material_id());
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space with at least one real per element component.
 */
CELER_FUNCTION Span<real_type> MaterialTrackView::element_scratch()
{
    auto offset = thread_.get() * params_.max_element_components;
    Span<real_type> all_scratch
        = states_.element_scratch[AllItems<real_type, MemSpace::native>{}];
    CELER_ENSURE(offset + params_.max_element_components <= all_scratch.size());
    return {all_scratch.data() + offset, params_.max_element_components};
}

//---------------------------------------------------------------------------//
/*!
 * Access the thread-local state.
 */
CELER_FORCEINLINE_FUNCTION MaterialTrackState& MaterialTrackView::state() const
{
    return states_.state[thread_];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
