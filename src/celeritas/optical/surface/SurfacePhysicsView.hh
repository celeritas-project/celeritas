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

    inline CELER_FUNCTION ActionId roughness_action_id() const;
    inline CELER_FUNCTION ActionId reflectivity_action_id() const;
    inline CELER_FUNCTION ActionId interaction_action_id() const;

    // Geometric surface normal
    inline CELER_FUNCTION Real3 const& surface_normal() const;

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
}  // namespace optical
}  // namespace celeritas
