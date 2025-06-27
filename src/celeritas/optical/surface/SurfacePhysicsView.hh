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
        SurfaceId surface;
        SurfaceLayerId layer;
    };

  public:
    // Construct from params, state, and surface ID for a given track
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             SurfaceId,
                                             TrackSlotId);

    // Initialize the boundary crossing for the track
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Get surface ID
    inline CELER_FUNCTION SurfaceId surface_id() const;

    inline CELER_FUNCTION SurfaceLayerId& current_layer();
    inline CELER_FUNCTION SurfaceLayerId current_layer() const;
    inline CELER_FUNCTION SurfaceLayerId::size_type num_layers() const;

    inline CELER_FUNCTION ActionId roughness_action_id() const;
    inline CELER_FUNCTION ActionId reflectivity_action_id() const;
    inline CELER_FUNCTION ActionId interaction_action_id() const;

    // Geometric surface normal
    inline CELER_FUNCTION Real3 const& surface_normal() const;

    // Normal of the current surface layer
    inline CELER_FUNCTION Real3 const& layer_normal() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
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
                                       TrackSlotId track_id)
    : params_(params), states_(states), surface_(surface), track_id_(track_id)
{
    CELER_EXPECT(track_id_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the boundary crossing for the track.
 */
CELER_FUNCTION SurfacePhysicsView&
SurfacePhysicsView::operator=(Initializer const& init)
{
    // TODO: Add assertions
    states_.surface_normal[track_id_] = init.normal;
    states_.surface[track_id_] = init.surface;
    states_.current_layer[track_id_] = init.layer;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal for the geometry's boundary.
 *
 * Defines the order in which the layers of the surface are traversed.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsView::surface_normal() const
{
    return states_.surface_normal[track_id_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
