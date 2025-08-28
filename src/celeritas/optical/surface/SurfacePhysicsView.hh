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

#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"

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
    //  Create view from surface physics data and state
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

    // Whether direction is exiting the boundary
    inline CELER_FUNCTION bool is_exiting(SubsurfaceDirection) const;

    // Whether the track is in the pre-volume
    inline CELER_FUNCTION bool in_pre_volume() const;

    // Whether the track is in the post-volume
    inline CELER_FUNCTION bool in_post_volume() const;

    //// ACCESS PHYSICS DATA ////

    // Position of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition subsurface_position() const;

    // Set position of the track in the surface crossing
    inline CELER_FUNCTION void subsurface_position(SurfaceTrackPosition);

    // Number of valid positions of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition::size_type num_positions() const;

    // Subsurface material of the current track position
    inline CELER_FUNCTION OptMatId subsurface_material() const;

    // Next subsurface interface in the given direction (track-local)
    inline CELER_FUNCTION
        PhysSurfaceId subsurface_interface(SubsurfaceDirection) const;

    // Calculate traversal direction from track momentum
    inline CELER_FUNCTION SubsurfaceDirection
    traversal_direction(Real3 const&) const;

    // Get surface model map for the given step and physics surface
    inline CELER_FUNCTION
        SurfacePhysicsMapView surface_physics_map(SurfacePhysicsOrder,
                                                  PhysSurfaceId) const;

    // Get local facet normal
    inline CELER_FUNCTION Real3 const& facet_normal() const;

    // Assign local facet normal
    inline CELER_FUNCTION void facet_normal(Real3 const&);

    //// MUTATORS ////

    // Cross subsurface interface in the given direction (track-local)
    inline CELER_FUNCTION void cross_subsurface_interface(SubsurfaceDirection);

    //// ACCESS SCALAR DATA ////

    // Default surface physics
    inline CELER_FUNCTION SurfaceId default_surface() const;

    // Get init-boundary action
    inline CELER_FUNCTION ActionId init_boundary_action() const;

    // Get surface stepping loop action
    inline CELER_FUNCTION ActionId surface_stepping_action() const;

    // Get post-boundary action
    inline CELER_FUNCTION ActionId post_boundary_action() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Get surface record of current geometric surface
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;

    // Get the record index from a track-local position
    template<class T, class U>
    CELER_FUNCTION U to_record_index(SurfaceTrackPosition,
                                     ItemMap<T, U> const&) const;
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
    CELER_EXPECT(init.surface);
    CELER_EXPECT(is_soft_unit_vector(init.global_normal));
    states_.surface[track_id_] = init.surface;
    states_.surface_orientation[track_id_] = init.orientation;
    states_.global_normal[track_id_] = init.global_normal;
    states_.facet_normal[track_id_] = init.global_normal;
    states_.surface_position[track_id_] = SurfaceTrackPosition{0};
    states_.pre_volume_material[track_id_] = init.pre_volume_material;
    states_.post_volume_material[track_id_] = init.post_volume_material;
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
    return static_cast<bool>(this->surface());
}

//---------------------------------------------------------------------------//
/*!
 * Whether the direction is exiting the surface.
 */
CELER_FUNCTION bool SurfacePhysicsView::is_exiting(SubsurfaceDirection d) const
{
    // Use unsigned underflow when moving reverse (-1) on the pre-surface
    // (position 0) to wrap to an invalid position value
    return static_cast<size_type>(this->subsurface_position().get()
                                  + to_signed_offset(d))
           >= this->num_positions();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is in the pre-volume.
 */
CELER_FUNCTION bool SurfacePhysicsView::in_pre_volume() const
{
    return this->subsurface_position().get() == 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is in the post-volume.
 */
CELER_FUNCTION bool SurfacePhysicsView::in_post_volume() const
{
    return this->subsurface_position().get() + 1 == this->num_positions();
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
CELER_FUNCTION SurfaceTrackPosition SurfacePhysicsView::subsurface_position() const
{
    return states_.surface_position[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Set current position of the track in the sub-surfaces, in track-local
 * coordinates.
 */
CELER_FUNCTION void
SurfacePhysicsView::subsurface_position(SurfaceTrackPosition pos)
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
 */
CELER_FUNCTION SurfaceTrackPosition::size_type
SurfacePhysicsView::num_positions() const
{
    return this->surface_record().subsurface_materials.size() + 2;
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface material ID of the current track position.
 */
CELER_FUNCTION OptMatId SurfacePhysicsView::subsurface_material() const
{
    if (this->in_pre_volume())
    {
        return states_.pre_volume_material[track_id_];
    }
    if (this->in_post_volume())
    {
        return states_.post_volume_material[track_id_];
    }

    CELER_ASSERT(this->subsurface_position().get() > 0);

    auto material_record_id
        = this->to_record_index(this->subsurface_position() - 1,
                                this->surface_record().subsurface_materials);
    CELER_ASSERT(material_record_id < params_.subsurface_materials.size());

    return params_.subsurface_materials[material_record_id];
}

//---------------------------------------------------------------------------//
/*!
 * Get the physics surface ID of the subsurface in the given direction.
 */
CELER_FUNCTION PhysSurfaceId
SurfacePhysicsView::subsurface_interface(SubsurfaceDirection d) const
{
    CELER_EXPECT(!this->is_exiting(d));

    auto track_pos = this->subsurface_position();
    if (d == SubsurfaceDirection::reverse)
    {
        --track_pos;
    }
    return this->to_record_index(track_pos,
                                 this->surface_record().subsurface_interfaces);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate traversal direction from track momentum.
 */
CELER_FUNCTION SubsurfaceDirection
SurfacePhysicsView::traversal_direction(Real3 const& dir) const
{
    CELER_EXPECT(is_soft_unit_vector(dir));
    return static_cast<SubsurfaceDirection>(
        is_entering_surface(dir, this->global_normal()));
}

//---------------------------------------------------------------------------//
/*!
 * Get surface model map for the given step and physics surface
 */
CELER_FUNCTION SurfacePhysicsMapView SurfacePhysicsView::surface_physics_map(
    SurfacePhysicsOrder step, PhysSurfaceId surface) const
{
    return SurfacePhysicsMapView{params_.model_maps[step], surface};
}

//---------------------------------------------------------------------------//
/*!
 * Get local facet normal after roughness sampling.
 */
CELER_FUNCTION Real3 const& SurfacePhysicsView::facet_normal() const
{
    return states_.facet_normal[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Assign local facet normal from roughness sampling.
 */
CELER_FUNCTION void SurfacePhysicsView::facet_normal(Real3 const& normal)
{
    CELER_EXPECT(is_soft_unit_vector(normal));
    states_.facet_normal[track_id_] = normal;
}

//---------------------------------------------------------------------------//
/*!
 * Cross the subsurface interface in the given direction.
 */
CELER_FUNCTION void
SurfacePhysicsView::cross_subsurface_interface(SubsurfaceDirection d)
{
    CELER_EXPECT(!this->is_exiting(d));
    this->subsurface_position(this->subsurface_position()
                              + to_signed_offset(d));
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
 * Get the surface record of the current geometric surface.
 */
CELER_FUNCTION SurfaceRecord const& SurfacePhysicsView::surface_record() const
{
    CELER_EXPECT(this->surface() < params_.surfaces.size());
    return params_.surfaces[this->surface()];
}

//---------------------------------------------------------------------------//
/*!
 * Convert track-local position to index in a surface record.
 */
template<class T, class U>
CELER_FUNCTION U SurfacePhysicsView::to_record_index(
    SurfaceTrackPosition pos, ItemMap<T, U> const& map) const
{
    T index{pos.get()};
    if (this->orientation() == SubsurfaceDirection::reverse)
    {
        index = T{map.size() - 1 - index.get()};
    }
    CELER_ASSERT(index < map.size());

    return map[index];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
