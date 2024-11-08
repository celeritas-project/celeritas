//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
using ValueGridId = OpaqueId<struct ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 */
class PhysicsTrackView
{
  public:
    using Energy = units::MevEnergy;

    inline CELER_FUNCTION PhysicsTrackView(OpticalMaterialId, TrackSlotId);

    inline CELER_FUNCTION void reset_interaction_mfp();
    inline CELER_FUNCTION real_type& interaction_mfp();
    inline CELER_FUNCTION real_type interaction_mfp() const;
    inline CELER_FUNCTION bool has_interaction_mfp() const;

    inline CELER_FUNCTION ModelId::size_type num_models() const;
    inline CELER_FUNCTION ActionId model_to_action(ModelId) const;
    inline CELER_FUNCTION ModelId action_to_model(ActionId) const;
    inline CELER_FUNCTION ActionId discrete_action() const;

    inline CELER_FUNCTION ValueGridId mfp_grid(ModelId) const;
    inline CELER_FUNCTION real_type calc_mfp(ModelId, Energy) const;
};

CELER_FUNCTION
PhysicsTrackView::PhysicsTrackView(OpticalMaterialId, TrackSlotId) {}

CELER_FUNCTION void PhysicsTrackView::reset_interaction_mfp() {}

CELER_FUNCTION real_type& PhysicsTrackView::interaction_mfp()
{
    static real_type x;
    return x;
}

CELER_FUNCTION real_type PhysicsTrackView::interaction_mfp() const
{
    return 0;
}

CELER_FUNCTION bool PhysicsTrackView::has_interaction_mfp() const
{
    return false;
}

CELER_FUNCTION ModelId::size_type PhysicsTrackView::num_models() const
{
    return 0;
}

CELER_FUNCTION ActionId PhysicsTrackView::model_to_action(ModelId) const
{
    return ActionId{};
}

CELER_FUNCTION ModelId PhysicsTrackView::action_to_model(ActionId) const
{
    return ModelId{};
}

CELER_FUNCTION ActionId PhysicsTrackView::discrete_action() const
{
    return ActionId{};
}

CELER_FUNCTION ValueGridId PhysicsTrackView::mfp_grid(ModelId) const
{
    return ValueGridId{};
}

CELER_FUNCTION real_type PhysicsTrackView::calc_mfp(ModelId, Energy) const
{
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
