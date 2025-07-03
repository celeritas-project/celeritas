//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

#include "SurfacePhysicsData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical surface physics data for a track.
 *
 * The surface physics view provides an interface for data and operations
 * used to manage an optical photon crossing a boundary. Tracks crossing
 * a boundary should be initialized first through the \c InitBoundaryAction
 * step.
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    using SurfaceStateRef = NativeRef<SurfacePhysicsStateData>;
    //!@}

    //! Data for initializing a track crossing a boundary
    struct Initializer
    {
        SurfaceId surface;
        Real3 normal;
    };

  public:
    // Construct from params, state, and surface ID for a given track
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             TrackSlotId);

    // Initialize the boundary crossing for the track
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Get surface ID
    inline CELER_FUNCTION SurfaceId surface_id() const;

    // Get roughness model for the surface
    inline CELER_FUNCTION ActionId roughness_action_id() const;

    // Get reflectivity model for the surface
    inline CELER_FUNCTION ActionId reflectivity_action_id() const;

    // Get interaction model for the surface
    inline CELER_FUNCTION ActionId interaction_action_id() const;

    // Geometric surface normal
    inline CELER_FUNCTION Real3 const& surface_normal() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Get the surface record
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;
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
    : params_(params), states_(states), track_id_(track_id)
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
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the normal for the geometry's boundary.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsView::surface_normal() const
{
    return states_.surface_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID of the surface the track is currently on.
 */
CELER_FUNCTION SurfaceId SurfacePhysicsView::surface_id() const
{
    return states_.surface[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for the roughness action of the surface.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::roughness_action_id() const
{
    return this->surface_record().roughness_model;
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for the reflectivity action of the surface.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::reflectivity_action_id() const
{
    return this->surface_record().reflectivity_model;
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for the interaction model of the surface.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::interaction_action_id() const
{
    return this->surface_record().interaction_model;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to access the surface record for this track.
 */
CELER_FUNCTION SurfaceRecord const& SurfacePhysicsView::surface_record() const
{
    CELER_EXPECT(this->surface_id() < params_.surfaces.size());
    return params_.surfaces[this->surface_id()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
