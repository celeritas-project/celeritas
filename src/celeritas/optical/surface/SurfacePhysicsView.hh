//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    using SurfaceStateRef = NativeCRef<SurfacePhysicsStateData>;
    //!@}

    //! Data for initializing a track crossing a boundary
    struct Initializer
    {
        Real3 normal;
    };

  public:
    // Construct from params, state, and surface ID for a given track
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             SurfaceId,
                                             TrackSlotId);

    // Initialize the boundary crossing for the track
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Get surface model for the current surface
    inline CELER_FUNCTION SurfaceModelId model() const;

    // Get surface ID
    inline CELER_FUNCTION SurfaceId surface_id() const;

    inline CELER_FUNCTION PerModelSurfaceId model_surface_id() const;

    inline CELER_FUNCTION void set_normal_action(ActionId);
    inline CELER_FUNCTION void set_calc_reflectivity_action(ActionId);
    inline CELER_FUNCTION void set_interaction_action(ActionId);

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    SurfaceId const surface_;
    TrackSlotId const track_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from params, state, and surface ID for a given track.
 */
CELER_FUNCTION
SurfacePhysicsView::SurfacePhysicsView(SurfaceParamsRef const& params,
                                       SurfaceStateRef const& states,
                                       SurfaceId surface,
                                       TrackSlotId track_id)
    : params_(params), states_(states), surface_(surface), track_id_(track_id)
{
    CELER_EXPECT(track_id_ < states_.size());
    CELER_EXPECT(surface_ < params_.scalars.num_surfaces);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the boundary crossing for the track.
 */
CELER_FUNCTION SurfacePhysicsView&
SurfacePhysicsView::operator=(Initializer const& init)
{
    states_.surface_normal[track_id_] = init.normal;
    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
