//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "MaterialData.hh"
#include "MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the material properties of a single particle track.
 *
 * These functions should be used in each physics Process or Interactor or
 * anything else that needs to access particle properties. Assume that all
 * these functions are expensive: when using them as accessors, locally store
 * the results rather than calling the function repeatedly. If any of the
 * calculations prove to be hot spots we will experiment with cacheing some of
 * the variables.
 *
 * The element scratch space is "thread-private" data with a fixed size
 * *greater than or equal to* the number of elemental components in the current
 * material.
 */
class MaterialTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t     = MaterialTrackState;
    using MaterialParamsRef = ::celeritas::NativeCRef<MaterialParamsData>;
    using MaterialStateRef  = ::celeritas::NativeRef<MaterialStateData>;
    //!@}

  public:
    // Construct from "static" parameters and "dynamic" state
    inline CELER_FUNCTION MaterialTrackView(const MaterialParamsRef& params,
                                            const MaterialStateRef&  states,
                                            ThreadId                 tid);

    // Initialize the particle
    inline CELER_FUNCTION MaterialTrackView&
    operator=(const Initializer_t& other);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Current material identifier
    CELER_FORCEINLINE_FUNCTION MaterialId material_id() const;

    //// STATIC PROPERTIES ////

    // Get a view to material properties
    CELER_FORCEINLINE_FUNCTION MaterialView make_material_view() const;

    // Access scratch space with at least one real per element component
    inline CELER_FUNCTION Span<real_type> element_scratch();

  private:
    const MaterialParamsRef& params_;
    const MaterialStateRef&  states_;
    const ThreadId           thread_;

    CELER_FORCEINLINE_FUNCTION MaterialTrackState& state() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
MaterialTrackView::MaterialTrackView(const MaterialParamsRef& params,
                                     const MaterialStateRef&  states,
                                     ThreadId                 tid)
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
CELER_FUNCTION MaterialId MaterialTrackView::material_id() const
{
    return this->state().material_id;
}

//---------------------------------------------------------------------------//
/*!
 * Get material properties for the current material.
 */
CELER_FUNCTION MaterialView MaterialTrackView::make_material_view() const
{
    return MaterialView(params_, this->material_id());
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space with at least one real per element component.
 */
CELER_FUNCTION Span<real_type> MaterialTrackView::element_scratch()
{
    auto            offset = thread_.get() * params_.max_element_components;
    Span<real_type> all_scratch
        = states_.element_scratch[AllItems<real_type, MemSpace::native>{}];
    CELER_ENSURE(offset + params_.max_element_components <= all_scratch.size());
    return all_scratch.subspan(offset, params_.max_element_components);
}

//---------------------------------------------------------------------------//
/*!
 * Access the thread-local state.
 */
CELER_FUNCTION MaterialTrackState& MaterialTrackView::state() const
{
    return states_.state[thread_];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
