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

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class PhysicsStepView
{
  public:
    inline CELER_FUNCTION PhysicsStepView(TrackSlotId);

    inline CELER_FUNCTION real_type& per_model_xs(ModelId mid);
    inline CELER_FUNCTION real_type per_model_xs(ModelId mid) const;
    inline CELER_FUNCTION real_type macro_xs() const;
    inline CELER_FUNCTION void macro_xs(real_type xs);
};

CELER_FUNCTION PhysicsStepView::PhysicsStepView(TrackSlotId) {}

CELER_FUNCTION real_type& PhysicsStepView::per_model_xs(ModelId)
{
    static real_type x;
    return x;
}

CELER_FUNCTION real_type PhysicsStepView::per_model_xs(ModelId) const
{
    return 0;
}

CELER_FUNCTION real_type PhysicsStepView::macro_xs() const
{
    return 0;
}

CELER_FUNCTION void PhysicsStepView::macro_xs(real_type) {}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
