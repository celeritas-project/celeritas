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
 *
 * During surface crossing, tracks maintain a "within-surface state" that
 * stores the current layer and direction, decoupling the geometry state from
 * the material state.
 *
 * See \c SurfacePhysicsView for documentation.
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
    inline CELER_FUNCTION bool is_exiting(LocalDirection) const;

    // Position of the track in the surface crossing
    inline CELER_FUNCTION LocalPositionId pos() const;

    // Next pos of the track in the given direction
    inline CELER_FUNCTION LocalPositionId next_pos() const;

    // Set position of the track in the surface crossing
    inline CELER_FUNCTION void pos(LocalPositionId);

    // Number of valid positions of the track in the surface crossing
    inline CELER_FUNCTION LocalSurfaceId::size_type num_local_pos() const;

    // Current track traversal direction
    inline CELER_FUNCTION LocalDirection dir() const;

    // Set track traversal direction
    inline CELER_FUNCTION void dir(LocalDirection);

    // Cross subsurface interface in the given direction
    inline CELER_FUNCTION void cross_interface(LocalDirection);

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
    states_.local_position[track_id_] = LocalPositionId{0};
    states_.local_direction[track_id_] = LocalDirection::forward;
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
    return this->pos().get() + 1 == this->num_local_pos();
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
CELER_FUNCTION bool SurfaceTraversalView::is_exiting(LocalDirection d) const
{
    // Use unsigned underflow when moving reverse (-1) on the pre-surface
    // (position 0) to wrap to an invalid position value
    return next_local_pos_id(this->pos(), d).unchecked_get()
           >= this->num_local_pos();
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
CELER_FUNCTION LocalPositionId SurfaceTraversalView::pos() const
{
    return states_.local_position[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION LocalPositionId SurfaceTraversalView::next_pos() const
{
    return next_local_pos_id(this->pos(), this->dir());
}

//---------------------------------------------------------------------------//
/*!
 * Set current position of the track in the sub-surfaces, in track-local
 * coordinates.
 */
CELER_FUNCTION void SurfaceTraversalView::pos(LocalPositionId pos)
{
    CELER_EXPECT(pos < this->num_local_pos());
    states_.local_position[track_id_] = pos;
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
CELER_FUNCTION LocalSurfaceId::size_type
SurfaceTraversalView::num_local_pos() const
{
    auto const& surface = params_.surfaces[states_.surface[track_id_]];
    return surface.interstitial_mat_ids.size() + 2;
}

//---------------------------------------------------------------------------//
/*!
 * Get current track traversal direction.
 *
 * This quantity is cached for a single loop of surface boundary crossing to
 * avoid repeated queries of the geometry. The traversal direction should be
 * updated when the geometry direction is updated after an interaction.
 */
CELER_FUNCTION LocalDirection SurfaceTraversalView::dir() const
{
    return states_.local_direction[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Set current track traversal direction.
 */
CELER_FUNCTION void SurfaceTraversalView::dir(LocalDirection dir)
{
    states_.local_direction[track_id_] = dir;
}

//---------------------------------------------------------------------------//
/*!
 * Cross the subsurface interface in the given direction.
 */
CELER_FUNCTION void SurfaceTraversalView::cross_interface(LocalDirection d)
{
    CELER_EXPECT(!this->is_exiting(d));
    this->pos(next_local_pos_id(this->pos(), d));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
