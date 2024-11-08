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

#include "PhysicsData.hh"
#include "Types.hh"

namespace celeritas
{
namespace optical
{
using ValueGridId = OpaqueId<struct ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 * Optical physics data for a track.
 *
 * The physics track data provides an interface for data and operations common
 * to most optical models. 
 */
class PhysicsTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from params, state, and material ID for a given track
    inline CELER_FUNCTION PhysicsTrackView(PhysicsParamsRef const&,
                                           PhysicsStateRef const&,
                                           OpticalMaterialId,
                                           TrackSlotId);

    //// Discrete interaction mean free path ////

    // Reset the currently calculated MFP
    inline CELER_FUNCTION void reset_interaction_mfp();

    // Get the current MFP
    inline CELER_FUNCTION real_type& interaction_mfp();

    // Get the current MFP
    inline CELER_FUNCTION real_type interaction_mfp() const;

    // Whether there's a currently calculated MFP
    inline CELER_FUNCTION bool has_interaction_mfp() const;

    //// Model-Action mappings ////

    // Number of optical models
    inline CELER_FUNCTION ModelId::size_type num_models() const;

    // Map a model ID to an action ID
    inline CELER_FUNCTION ActionId model_to_action(ModelId) const;

    // Map an action ID to a model ID
    inline CELER_FUNCTION ModelId action_to_model(ActionId) const;

    // ID of the discrete action
    inline CELER_FUNCTION ActionId discrete_action() const;

    //// Mean free path grids ////

    // Get MFP grid ID for the given model
    inline CELER_FUNCTION ValueGridId mfp_grid(ModelId) const;

    // Calculate the MFP for the given model and energy
    inline CELER_FUNCTION real_type calc_mfp(ModelId, Energy) const;

  private:
    PhysicsParamsRef const& params_;
    PhysicsStateRef const& states_;
    OpticalMaterialId const opt_material_;
    TrackSlotId const track_id_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from params, state, and material ID for a given track.
 */
CELER_FUNCTION
PhysicsTrackView::PhysicsTrackView(PhysicsParamsRef const& params,
                                   PhysicsStateRef const& states,
                                   OpticalMaterialId opt_mat,
                                   TrackSlotId track_id)
    : params_(params)
    , states_(states)
    , opt_material_(opt_mat)
    , track_id_(track_id)
{
    CELER_EXPECT(track_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Reset the currently calculated interaction MFP.
 */
CELER_FUNCTION void PhysicsTrackView::reset_interaction_mfp() {}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the interaction mean free path.
 */
CELER_FUNCTION real_type& PhysicsTrackView::interaction_mfp()
{
    static real_type x;
    return x;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the interaction mean free path.
 */
CELER_FUNCTION real_type PhysicsTrackView::interaction_mfp() const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether there's a calculated interaction MFP.
 */
CELER_FUNCTION bool PhysicsTrackView::has_interaction_mfp() const
{
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the number of optical models.
 */
CELER_FUNCTION ModelId::size_type PhysicsTrackView::num_models() const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a model ID to an action ID.
 */
CELER_FUNCTION ActionId PhysicsTrackView::model_to_action(ModelId) const
{
    return ActionId{};
}

//---------------------------------------------------------------------------//
/*!
 * Convert an action ID to a model ID.
 */
CELER_FUNCTION ModelId PhysicsTrackView::action_to_model(ActionId) const
{
    return ModelId{};
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for the discrete action.
 */
CELER_FUNCTION ActionId PhysicsTrackView::discrete_action() const
{
    return ActionId{};
}

//---------------------------------------------------------------------------//
/*!
 * Get the MFP grid ID for the given model.
 *
 * The grid corresponds to the optical material this track view was
 * constructed with.
 */
CELER_FUNCTION ValueGridId PhysicsTrackView::mfp_grid(ModelId) const
{
    return ValueGridId{};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the MFP for the given model and energy.
 *
 * Energy is interpolated using \c GenericGridCalculator for the model's
 * MFP grid.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_mfp(ModelId, Energy) const
{
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
