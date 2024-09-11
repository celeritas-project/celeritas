//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    PhysicsTrackView ...;
   \endcode
 */
class PhysicsTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    //!@}

  public:
    inline CELER_FUNCTION PhysicsTrackView(PhysicsParamsRef const& params,
                                           PhysicsStateRef const& states,
                                           OpticalMaterialId opt_material,
                                           TrackSlotId tid);

    CELER_FORCEINLINE_FUNCTION OpticalMaterialId optical_material_id() const
    {
        return opt_material_;
    }

    CELER_FORCEINLINE_FUNCTION PhysicsParamsScalars const& scalars() const
    {
        return params_.scalars;
    }

    CELER_FORCEINLINE_FUNCTION size_type num_optical_models() const
    {
        return params_.model_tables.size();
    }

    inline CELER_FUNCTION ValueGridId mfp_grid(OpticalModelId) const;

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

CELER_FUNCTION auto PhysicsTrackView::mfp_grid(OpticalModelId mid) const
    -> ValueGridId
{
    CELER_EXPECT(mid < this->num_optical_models());

    ValueTableId table_id = params_.models[mid.get()].mfp_table;
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
// IMPLEMENTATION HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Get the thread-local state (mutable)
CELER_FUNCTION PhysicsTrackState& PhysicsTrackView::state()
{
    return state_.state[track_slot_];
}

//! Get the thread-local state (const)
CELER_FUNCTION PhysicsTrackState const& PhysicsTrackView::state() const
{
    return state_.state[track_slot_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
