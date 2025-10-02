//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/phys/SurfacePhysicsMapView.hh"

#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"
#include "SurfacePhysicsView.hh"
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
class SurfacePhysicsTrackView
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
    inline CELER_FUNCTION SurfacePhysicsTrackView(SurfaceParamsRef const&,
                                                  SurfaceStateRef const&,
                                                  TrackSlotId);

    // Initialize track state
    inline CELER_FUNCTION SurfacePhysicsTrackView&
    operator=(Initializer const&);

    // Reset surface physics state of the track
    inline CELER_FUNCTION void reset();

    // Whether track is undergoing boundary crossing
    inline CELER_FUNCTION bool is_crossing_boundary() const;

    // Get current subsurface material
    inline CELER_FUNCTION OptMatId material() const;

    // Get next subsurface material
    inline CELER_FUNCTION OptMatId next_material() const;

    // Get surface physics map for next surface
    inline CELER_FUNCTION
        SurfacePhysicsMapView interface(SurfacePhysicsOrder) const;

    // Calculate and update traversal direction from track momentum
    inline CELER_FUNCTION void update_traversal_direction(Real3 const&);

    // Get global surface normal
    inline CELER_FUNCTION Real3 const& global_normal() const;

    // Get local facet normal
    inline CELER_FUNCTION Real3 const& facet_normal() const;

    // Assign local facet normal
    inline CELER_FUNCTION void facet_normal(Real3 const&);

    // Construct a traversal view for this track
    inline CELER_FUNCTION SurfaceTraversalView traversal() const;

    // Construct a constant physics view
    inline CELER_FUNCTION SurfacePhysicsView surface() const;

    // Access scalar data for surface physics
    inline CELER_FUNCTION SurfacePhysicsParamsScalars const& scalars() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Get material at the given track position
    inline CELER_FUNCTION OptMatId material(SurfaceTrackPosition) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize view from surface physics data and state for a given track.
 */
CELER_FUNCTION
SurfacePhysicsTrackView::SurfacePhysicsTrackView(SurfaceParamsRef const& params,
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
CELER_FUNCTION SurfacePhysicsTrackView&
SurfacePhysicsTrackView::operator=(Initializer const& init)
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
CELER_FUNCTION void SurfacePhysicsTrackView::reset()
{
    states_.surface[track_id_] = {};
    CELER_ENSURE(!states_.surface[track_id_]);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is undergoing boundary crossing.
 *
 * Returns true if there's a valid surface ID, otherwise false.
 */
CELER_FUNCTION bool SurfacePhysicsTrackView::is_crossing_boundary() const
{
    return states_.surface[track_id_] < params_.surfaces.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get current subsurface material.
 */
CELER_FUNCTION OptMatId SurfacePhysicsTrackView::material() const
{
    return this->material(this->traversal().pos());
}

//---------------------------------------------------------------------------//
/*!
 * Get next subsurface material.
 */
CELER_FUNCTION OptMatId SurfacePhysicsTrackView::next_material() const
{
    return this->material(this->traversal().next_pos());
}

//---------------------------------------------------------------------------//
/*!
 * Get surface physics map for next surface.
 */
CELER_FUNCTION SurfacePhysicsMapView
SurfacePhysicsTrackView::interface(SurfacePhysicsOrder step) const
{
    auto traverse = this->traversal();
    return SurfacePhysicsMapView{
        params_.model_maps[step],
        this->surface().interface(traverse.pos(), traverse.dir())};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate and update traversal direction from track momentum.
 */
CELER_FUNCTION void
SurfacePhysicsTrackView::update_traversal_direction(Real3 const& dir)
{
    CELER_EXPECT(is_soft_unit_vector(dir));
    this->traversal().dir(
        calc_subsurface_direction(dir, this->global_normal()));
}

//---------------------------------------------------------------------------//
/*!
 * Get global surface normal.
 *
 * The global surface normal is the normal defined by the geometry and does not
 * include any roughness effects. By convention it points from the post-volume
 * into the pre-volume.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsTrackView::global_normal() const
{
    CELER_EXPECT(this->is_crossing_boundary());
    return states_.global_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Get local facet normal after roughness sampling.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsTrackView::facet_normal() const
{
    CELER_EXPECT(this->is_crossing_boundary());
    return states_.facet_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Assign local facet normal from roughness sampling.
 */
CELER_FUNCTION void SurfacePhysicsTrackView::facet_normal(Real3 const& normal)
{
    CELER_EXPECT(this->is_crossing_boundary());
    CELER_EXPECT(is_soft_unit_vector(normal));
    states_.facet_normal[track_id_] = normal;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a traversal view for this track.
 */
CELER_FUNCTION SurfaceTraversalView SurfacePhysicsTrackView::traversal() const
{
    return SurfaceTraversalView{params_, states_, track_id_};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a surface physics view, whose orientation is based on this track.
 */
CELER_FUNCTION SurfacePhysicsView SurfacePhysicsTrackView::surface() const
{
    return SurfacePhysicsView{params_,
                              states_.surface[track_id_],
                              states_.surface_orientation[track_id_]};
}

//---------------------------------------------------------------------------//
/*!
 * Access scalar data for surface physics.
 */
CELER_FUNCTION SurfacePhysicsParamsScalars const&
SurfacePhysicsTrackView::scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
/*!
 * Get material at given track position.
 */
CELER_FUNCTION OptMatId
SurfacePhysicsTrackView::material(SurfaceTrackPosition pos) const
{
    auto pos_range
        = range(SurfaceTrackPosition{this->traversal().num_positions()});
    CELER_EXPECT(pos < pos_range.size());

    if (pos == pos_range.front())
    {
        // Pre-volume material
        return states_.pre_volume_material[track_id_];
    }
    else if (pos == pos_range.back())
    {
        // Post-volume material
        return states_.post_volume_material[track_id_];
    }

    return this->surface().interstitial_material(pos);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
