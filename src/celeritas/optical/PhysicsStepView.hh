//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "PhysicsData.hh"
#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access step-local (non-persistent) optical physics track data.
 */
class PhysicsStepView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    //!@}

  public:
    // Construct from state data for a given track
    inline CELER_FUNCTION PhysicsStepView(PhysicsStateRef const&, TrackSlotId);

    //// Cross section scrach space ////

    // Set cross section for a given model
    inline CELER_FUNCTION real_type& per_model_xs(ModelId mid);

    // Set total cross section
    inline CELER_FUNCTION void macro_xs(real_type xs);

    // Retrieve cross section for a given model
    inline CELER_FUNCTION real_type per_model_xs(ModelId mid) const;

    // Retrieve total cross section
    inline CELER_FUNCTION real_type macro_xs() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state data for a given track.
 */
CELER_FUNCTION
PhysicsStepView::PhysicsStepView(PhysicsStateRef const&, TrackSlotId) {}

//---------------------------------------------------------------------------//
/*!
 * Set cross section for a given model.
 */
CELER_FUNCTION real_type& PhysicsStepView::per_model_xs(ModelId)
{
    static real_type x;
    return x;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve cross section for a given model.
 */
CELER_FUNCTION real_type PhysicsStepView::per_model_xs(ModelId) const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve total cross section.
 */
CELER_FUNCTION real_type PhysicsStepView::macro_xs() const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Set total cross section.
 */
CELER_FUNCTION void PhysicsStepView::macro_xs(real_type) {}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
