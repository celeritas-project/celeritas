//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/phys/SurfacePhysicsMapView.hh"

#include "SurfaceModelView.hh"
#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"
#include "SurfaceRecordView.hh"
#include "SurfaceTraversalView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical surface physics data for a track.
 *
 * Tracks maintain a position while traversing the interstitial materials of an
 * optical surface. This class provides transformations from this position
 * based on the surface orientation and traversal direction to access relevant
 * material and interface data in storage.
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    using SurfaceStateRef = NativeRef<SurfacePhysicsStateData>;
    //!@}

    struct Initializer
    {
        SurfaceId surface{};
        SubsurfaceDirection orientation;
        Real3 global_normal{0, 0, 0};
        OptMatId pre_volume_material{};
        OptMatId post_volume_material{};
    };

  public:
    // Create view from surface physics data and state
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             TrackSlotId);

    //// INITIALIZATION ////

    // Initialize track state
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Reset surface physics state of the track
    inline CELER_FUNCTION void reset();

    //// STATE INVARIANTS ////

    // Get current geometric surface
    inline CELER_FUNCTION SurfaceId surface() const;

    // Get surface orientation
    inline CELER_FUNCTION SubsurfaceDirection orientation() const;

    // Get global surface normal
    inline CELER_FUNCTION Real3 const& global_normal() const;

    //// QUERY CROSSING STATE ////

    // Whether track is undergoing boundary crossing
    inline CELER_FUNCTION bool is_crossing_boundary() const;

    //// ACCESS PHYSICS DATA ////

    // Calculate and update traversal direction from track momentum
    inline CELER_FUNCTION void traversal_direction(Real3 const&);

    // Get surface model for the given step
    inline CELER_FUNCTION
        SurfaceModelView surface_model(SurfacePhysicsOrder) const;

    // Get local facet normal
    inline CELER_FUNCTION Real3 const& facet_normal() const;

    // Assign local facet normal
    inline CELER_FUNCTION void facet_normal(Real3 const&);

    //// ACCESS SCALAR DATA ////

    // Default surface physics
    inline CELER_FUNCTION SurfaceId default_surface() const;

    // Get init-boundary action
    inline CELER_FUNCTION ActionId init_boundary_action() const;

    // Get surface stepping loop action
    inline CELER_FUNCTION ActionId surface_stepping_action() const;

    // Get post-boundary action
    inline CELER_FUNCTION ActionId post_boundary_action() const;

    // Construct a traversal view for this track
    inline CELER_FUNCTION SurfaceTraversalView traversal() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Subsurface material at the given position
    inline CELER_FUNCTION
        OptMatId subsurface_material(SurfaceTrackPosition) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize view from surface physics data and state for a given track.
 */
CELER_FUNCTION
SurfacePhysicsView::SurfacePhysicsView(SurfaceParamsRef const& params,
                                       SurfaceStateRef const& states,
                                       TrackSlotId track)
    : params_(params), states_(states), track_id_(track)
{
    CELER_EXPECT(track_id_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track state with given initializer data.
 */
CELER_FUNCTION SurfacePhysicsView&
SurfacePhysicsView::operator=(Initializer const& init)
{
    CELER_EXPECT(init.surface < params_.surfaces.size());
    CELER_EXPECT(is_soft_unit_vector(init.global_normal));
    states_.surface[track_id_] = init.surface;
    states_.surface_orientation[track_id_] = init.orientation;
    states_.global_normal[track_id_] = init.global_normal;
    states_.facet_normal[track_id_] = init.global_normal;
    states_.pre_volume_material[track_id_] = init.pre_volume_material;
    states_.post_volume_material[track_id_] = init.post_volume_material;
    this->traversal() = SurfaceTraversalView::Initializer{};
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the state of a track.
 *
 * Invalidates the surface ID, indicating the track is no longer undergoing
 * boundary crossing.
 */
CELER_FUNCTION void SurfacePhysicsView::reset()
{
    states_.surface[track_id_] = {};
    CELER_ENSURE(!states_.surface[track_id_]);
}

//---------------------------------------------------------------------------//
/*!
 * Get geometric surface ID the track is currently on.
 *
 * The ID is invalid if the track is not undergoing a boundary crossing.
 */
CELER_FUNCTION SurfaceId SurfacePhysicsView::surface() const
{
    return states_.surface[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Get traversal orientation of the current surface.
 *
 * Subsurfaces are ordered in storage between two volumes. This orientation
 * specifies if the track is traversing the stored list of sub-surfaces in
 * forward or reverse order.
 */
CELER_FUNCTION SubsurfaceDirection SurfacePhysicsView::orientation() const
{
    CELER_EXPECT(this->is_crossing_boundary());
    return states_.surface_orientation[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Get global surface normal.
 *
 * The global surface normal is the normal defined by the geometry and does not
 * include any roughness effects. By convention it points from the post-volume
 * into the pre-volume.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsView::global_normal() const
{
    CELER_EXPECT(this->is_crossing_boundary());
    return states_.global_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is undergoing boundary crossing.
 *
 * Returns true if there's a valid surface ID, otherwise false.
 */
CELER_FUNCTION bool SurfacePhysicsView::is_crossing_boundary() const
{
    return this->surface() < params_.surfaces.size();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate and update traversal direction from track momentum.
 */
CELER_FUNCTION void SurfacePhysicsView::traversal_direction(Real3 const& dir)
{
    CELER_EXPECT(is_soft_unit_vector(dir));
    this->traversal().direction(static_cast<SubsurfaceDirection>(
        is_entering_surface(dir, this->global_normal())));
}

//---------------------------------------------------------------------------//
/*!
 * Get surface model view of the given step in the given direction.
 */
CELER_FUNCTION SurfaceModelView
SurfacePhysicsView::surface_model(SurfacePhysicsOrder step) const
{
    CELER_EXPECT(step != SurfacePhysicsOrder::size_);

    auto traverse = this->traversal();
    CELER_ASSERT(!traverse.is_exiting());

    auto phys_surface
        = SurfaceRecordView{params_, this->surface(), this->orientation()}
              .interface(traverse.position(), traverse.direction());
    CELER_ASSERT(phys_surface);

    return SurfaceModelView{
        SurfacePhysicsMapView{params_.model_maps[step], phys_surface},
        this->subsurface_material(traverse.position()),
        this->subsurface_material(traverse.next_position())};
}

//---------------------------------------------------------------------------//
/*!
 * Get local facet normal after roughness sampling.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsView::facet_normal() const
{
    CELER_EXPECT(this->is_crossing_boundary());
    return states_.facet_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Assign local facet normal from roughness sampling.
 */
CELER_FUNCTION void SurfacePhysicsView::facet_normal(Real3 const& normal)
{
    CELER_EXPECT(this->is_crossing_boundary());
    CELER_EXPECT(is_soft_unit_vector(normal));
    states_.facet_normal[track_id_] = normal;
}

//---------------------------------------------------------------------------//
/*!
 * Get the default surface.
 */
CELER_FUNCTION SurfaceId SurfacePhysicsView::default_surface() const
{
    return params_.scalars.default_surface;
}

//---------------------------------------------------------------------------//
/*!
 * Get init-boundary action.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::init_boundary_action() const
{
    return params_.scalars.init_boundary_action;
}

//---------------------------------------------------------------------------//
/*!
 * Get surface stepping loop action.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::surface_stepping_action() const
{
    return params_.scalars.surface_stepping_action;
}

//---------------------------------------------------------------------------//
/*!
 * Get post-boundary action.
 */
CELER_FUNCTION ActionId SurfacePhysicsView::post_boundary_action() const
{
    return params_.scalars.post_boundary_action;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a traversal view for this track.
 */
CELER_FUNCTION SurfaceTraversalView SurfacePhysicsView::traversal() const
{
    return SurfaceTraversalView{params_, states_, track_id_};
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface material ID of the current track position.
 */
CELER_FUNCTION OptMatId
SurfacePhysicsView::subsurface_material(SurfaceTrackPosition pos) const
{
    CELER_EXPECT(this->is_crossing_boundary());

    auto pos_range
        = range(SurfaceTrackPosition{this->traversal().num_positions()});

    if (pos == pos_range.front())
    {
        // In pre-volume
        return states_.pre_volume_material[track_id_];
    }
    if (pos == pos_range.back())
    {
        // In post-volume
        return states_.post_volume_material[track_id_];
    }

    return SurfaceRecordView{params_, this->surface(), this->orientation()}
        .material(pos);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
