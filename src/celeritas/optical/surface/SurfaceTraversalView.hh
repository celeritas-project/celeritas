//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceTraversalView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/optical/Types.hh"

#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Manage one-dimensional logic for traversing an optical surface.
 */
class SurfaceTraversalView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    using SurfaceStateRef = NativeRef<SurfacePhysicsStateData>;
    //!@}

    struct Initializer
    {
    };

  public:
    // Create view from surface physics data and state
    inline CELER_FUNCTION SurfaceTraversalView(SurfaceParamsRef const&,
                                               SurfaceStateRef const&,
                                               TrackSlotId);

    // Initialize track state
    inline CELER_FUNCTION SurfaceTraversalView& operator=(Initializer const&);

    // Whether the track is in the pre-volume
    inline CELER_FUNCTION bool in_pre_volume() const;

    // Whether the track is in the post-volume
    inline CELER_FUNCTION bool in_post_volume() const;

    // Whether track is currently moving off the surface
    inline CELER_FUNCTION bool is_exiting() const;

    // Whether track is in pre-/post-volume and direction is off the surface
    inline CELER_FUNCTION bool is_exiting(SubsurfaceDirection) const;

    // Position of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition pos() const;

    // Next pos of the track in the given direction
    inline CELER_FUNCTION SurfaceTrackPosition next_pos() const;

    // Set position of the track in the surface crossing
    inline CELER_FUNCTION void pos(SurfaceTrackPosition);

    // Number of valid positions of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition::size_type num_positions() const;

    // Current track traversal direction
    inline CELER_FUNCTION SubsurfaceDirection dir() const;

    // Set track traversal direction
    inline CELER_FUNCTION void dir(SubsurfaceDirection);

    // Cross subsurface interface in the given direction
    inline CELER_FUNCTION void cross_interface(SubsurfaceDirection);

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create view from surface physics data and state for a given track.
 */
CELER_FUNCTION
SurfaceTraversalView::SurfaceTraversalView(SurfaceParamsRef const& params,
                                           SurfaceStateRef const& states,
                                           TrackSlotId track)
    : params_(params), states_(states), track_id_(track)
{
    CELER_EXPECT(track_id_ < states_.size());
    CELER_EXPECT(states_.surface[track_id_] < params_.surfaces.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track state with given initializer data.
 */
CELER_FUNCTION SurfaceTraversalView&
SurfaceTraversalView::operator=(Initializer const&)
{
    states_.surface_position[track_id_] = SurfaceTrackPosition{0};
    states_.track_direction[track_id_] = SubsurfaceDirection::forward;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is in the pre-volume.
 */
CELER_FUNCTION bool SurfaceTraversalView::in_pre_volume() const
{
    return this->pos().get() == 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is in the post-volume.
 */
CELER_FUNCTION bool SurfaceTraversalView::in_post_volume() const
{
    return this->pos().get() + 1 == this->num_positions();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the current track direction is exiting the surface.
 */
CELER_FUNCTION bool SurfaceTraversalView::is_exiting() const
{
    return this->is_exiting(this->dir());
}

//---------------------------------------------------------------------------//
/*!
 * Whether the direction is exiting the surface.
 */
CELER_FUNCTION bool
SurfaceTraversalView::is_exiting(SubsurfaceDirection d) const
{
    // Use unsigned underflow when moving reverse (-1) on the pre-surface
    // (position 0) to wrap to an invalid position value
    return next_subsurface_position(this->pos(), d).unchecked_get()
           >= this->num_positions();
}

//---------------------------------------------------------------------------//
/*!
 * Current position of the track in the sub-surfaces, in track-local
 * coordinates.
 *
 * Tracks traverse a surface in track-local coordinates where 0 is the
 * pre-volume and N is the post-volume. Depending on the surface orientation,
 * this will be mapped to the appropriate sub-surface material and interface.
 */
CELER_FUNCTION SurfaceTrackPosition SurfaceTraversalView::pos() const
{
    return states_.surface_position[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION SurfaceTrackPosition SurfaceTraversalView::next_pos() const
{
    return next_subsurface_position(this->pos(), this->dir());
}

//---------------------------------------------------------------------------//
/*!
 * Set current position of the track in the sub-surfaces, in track-local
 * coordinates.
 */
CELER_FUNCTION void SurfaceTraversalView::pos(SurfaceTrackPosition pos)
{
    CELER_EXPECT(pos < this->num_positions());
    states_.surface_position[track_id_] = pos;
}

//---------------------------------------------------------------------------//
/*!
 * Get number of valid track positions in the surface.
 *
 * This is equivalent to the number of interstitial sub-surface materials, plus
 * the pre-volume and post-volumes.
 *
 * \todo Check if caching this would improve performance over redirections.
 */
CELER_FUNCTION SurfaceTrackPosition::size_type
SurfaceTraversalView::num_positions() const
{
    return params_.surfaces[states_.surface[track_id_]]
               .subsurface_materials.size()
           + 2;
}

//---------------------------------------------------------------------------//
/*!
 * Get current track traversal direction.
 *
 * This quantity is cached for a single loop of surface boundary crossing to
 * avoid repeated queries of the geometry. The traversal direction should be
 * updated when the geometry direction is updated after an interaction.
 */
CELER_FUNCTION SubsurfaceDirection SurfaceTraversalView::dir() const
{
    return states_.track_direction[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Set current track traversal direction.
 */
CELER_FUNCTION void SurfaceTraversalView::dir(SubsurfaceDirection dir)
{
    states_.track_direction[track_id_] = dir;
}

//---------------------------------------------------------------------------//
/*!
 * Cross the subsurface interface in the given direction.
 */
CELER_FUNCTION void SurfaceTraversalView::cross_interface(SubsurfaceDirection d)
{
    CELER_EXPECT(!this->is_exiting(d));
    this->pos(next_subsurface_position(this->pos(), d));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
