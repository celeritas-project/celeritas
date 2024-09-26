//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"

#include "PhysicsData.hh"

namespace celeritas
{
namespace optical
{
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
    using Initializer_t = PhysicsTrackInitializer;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from params, state, and material ID for the given track
    inline CELER_FUNCTION PhysicsTrackView(PhysicsParamsRef const& params,
                                           PhysicsStateRef const& states,
                                           OpticalMaterialId opt_material,
                                           TrackSlotId tid);

    //// MUTATORS ////

    // Initialize the view
    CELER_FORCEINLINE_FUNCTION PhysicsTrackView&
    operator=(Initializer_t const&);

    // Set the remaining interaction MFP distance
    inline CELER_FUNCTION void interaction_mfp(real_type);

    // Unassign the remaining interaction MFP distance
    inline CELER_FUNCTION void reset_interaction_mfp();

    //// FREE ACCESSORS ////

    // Whether the remaining interaction MFP has been calculated
    CELER_FORCEINLINE_FUNCTION bool has_interaction_mfp() const;

    // Remaining interaction MFP distance
    CELER_FORCEINLINE_FUNCTION real_type interaction_mfp() const;

    // Current material ID
    CELER_FORCEINLINE_FUNCTION OpticalMaterialId optical_material_id() const;

    // Number of active optical models
    CELER_FORCEINLINE_FUNCTION size_type num_optical_models() const;

    //// CALCULATORS ////

    // Calculate the MFP for the given model and energy
    inline CELER_FUNCTION real_type calc_mfp(ModelId, Energy) const;

    // Retrieve the energy grid ID for the given model
    inline CELER_FUNCTION ValueGridId mfp_grid(ModelId) const;

    // Construct a calculator for the given grid ID
    template<class Calc>
    inline CELER_FUNCTION Calc make_calculator(ValueGridId id) const;

    //// PARAMETER DATA ////

    // Map an action ID to a model ID
    inline CELER_FUNCTION ModelId action_to_model(ActionId) const;

    // Map a model ID to an action ID
    inline CELER_FUNCTION ActionId model_to_action(ModelId) const;

    // Physics scalar parameters
    CELER_FORCEINLINE_FUNCTION PhysicsParamsScalars const& scalars() const;

  private:
    PhysicsParamsRef const& params_;
    PhysicsStateRef const& states_;
    OpticalMaterialId const opt_material_;
    TrackSlotId const track_slot_;

    //// IMPLEMENTATION HELPER FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION PhysicsTrackState const& state() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data.
 */
CELER_FUNCTION
PhysicsTrackView::PhysicsTrackView(PhysicsParamsRef const& params,
                                   PhysicsStateRef const& states,
                                   OpticalMaterialId opt_material,
                                   TrackSlotId tid)
    : params_(params)
    , states_(states)
    , opt_material_(opt_material)
    , track_slot_(tid)
{
    CELER_EXPECT(track_slot_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the track view.
 */
CELER_FUNCTION PhysicsTrackView&
PhysicsTrackView::operator=(Initializer_t const&)
{
    this->state().interaction_mfp = 0;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Set the remaining interaction MFP distance.
 *
 * This value is decremented every step the photon moves.
 */
CELER_FUNCTION void PhysicsTrackView::interaction_mfp(real_type mfp)
{
    CELER_EXPECT(mfp > 0);
    this->state().interaction_mfp = mfp;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the remaining interaction MFP distance.
 *
 * This value is decremented every step the photon moves.
 */
CELER_FUNCTION void PhysicsTrackView::reset_interaction_mfp()
{
    this->state().interaction_mfp = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the remaining interaction MFP distance has been calculated.
 */
CELER_FUNCTION bool PhysicsTrackView::has_interaction_mfp() const
{
    return this->state().interaction_mfp > 0;
}

//---------------------------------------------------------------------------//
/*!
 * The remaining interaction MFP distance.
 */
CELER_FUNCTION real_type PhysicsTrackView::interaction_mfp() const
{
    return this->state().interaction_mfp;
}

//---------------------------------------------------------------------------//
/*!
 * The current optical material identifier.
 */
CELER_FUNCTION OpticalMaterialId PhysicsTrackView::optical_material_id() const
{
    return opt_material_;
}

//---------------------------------------------------------------------------//
/*!
 * The number of active optical models.
 */
CELER_FUNCTION size_type PhysicsTrackView::num_optical_models() const
{
    return params_.scalars.num_models;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the macroscopic cross section for the given model and energy.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_mfp(ModelId mid,
                                                    Energy energy) const
{
    real_type result = 0;

    if (auto grid_id = this->mfp_grid(mid))
    {
        auto calc = this->make_calculator<GenericCalculator>(grid_id);
        result = calc(value_as<Energy>(energy));
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the MFP grid identifier for the given model in the current
 * material.
 */
CELER_FUNCTION auto PhysicsTrackView::mfp_grid(ModelId mid) const -> ValueGridId
{
    CELER_EXPECT(mid < this->num_optical_models());

    auto table_id = params_.model_tables[mid].mfp_table;
    CELER_ASSERT(table_id);

    ValueTable const& table = params_.tables[table_id];
    if (!table)
        return {};  // no table for this model?

    CELER_EXPECT(opt_material_ < table.grids.size());
    auto grid_id_ref = table.grids[opt_material_.get()];
    if (!grid_id_ref)
        return {};  // no table for this material

    return params_.grid_ids[grid_id_ref];
}

//---------------------------------------------------------------------------//
/*!
 * Construct a calculator for the given grid identifier.
 */
template<class Calc>
CELER_FUNCTION Calc PhysicsTrackView::make_calculator(ValueGridId id) const
{
    CELER_ASSERT(id < params_.grids.size());
    return Calc{params_.grids[id], params_.reals};
}

//---------------------------------------------------------------------------//
/*!
 * Map a model ID to an action ID.
 */
CELER_FUNCTION ModelId PhysicsTrackView::action_to_model(ActionId aid) const
{
    if (!aid)
        return ModelId{};

    // Rely on unsigned rollover if action ID is less than the first model
    ModelId::size_type result = aid.unchecked_get()
                                - this->scalars().model_to_action;
    if (result >= this->num_optical_models())
        return ModelId{};

    return ModelId{result};
}

//---------------------------------------------------------------------------//
/*!
 * Map a model ID to an action ID.
 */
CELER_FUNCTION ActionId PhysicsTrackView::model_to_action(ModelId mid) const
{
    CELER_ASSERT(mid < this->num_optical_models());
    return ActionId{mid.unchecked_get() + this->scalars().model_to_action};
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the physics scalar parameters.
 */
CELER_FUNCTION PhysicsParamsScalars const& PhysicsTrackView::scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Get the thread-local state (mutable)
CELER_FUNCTION PhysicsTrackState& PhysicsTrackView::state()
{
    return states_.states[track_slot_];
}

//! Get the thread-local state (const)
CELER_FUNCTION PhysicsTrackState const& PhysicsTrackView::state() const
{
    return states_.states[track_slot_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
