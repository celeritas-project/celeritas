//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/Types.hh"

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

    struct Initializer
    {
        GeometricSurfaceId surface;
        SubsurfaceDirection orientation;
    };

  public:
    //  Create view from surface physics data and state
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             TrackSlotId);

    // Initialize track state
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Reset surface physics state of the track
    inline CELER_FUNCTION void reset() const;

    // Get current geometric surface
    inline CELER_FUNCTION GeometricSurfaceId surface() const;

    // Get surface orientation
    inline CELER_FUNCTION SubsurfaceDirection orientation() const;

    // Whether track is undergoing boundary crossing
    inline CELER_FUNCTION bool is_crossing_boundary() const;

    // Whether the track is in the pre-volume
    inline CELER_FUNCTION bool in_pre_volume() const;

    // Whether the track is in the post-volume
    inline CELER_FUNCTION bool in_post_volume() const;

    // Position of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition subsurface_position() const;

    // Position of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition& subsurface_position();

    // Number of valid positions of the track in the surface crossing
    inline CELER_FUNCTION SurfaceTrackPosition::size_type num_positions() const;

    // Subsurface material of the current track position
    inline CELER_FUNCTION OptMatId subsurface_material() const;

    // Next subsurface interface in the given direction (track-local)
    inline CELER_FUNCTION
        PhysicsSurfaceId subsurface_interface(SubsurfaceDirection) const;

    // Cross subsurface interface in the given direction (track-local)
    inline CELER_FUNCTION void cross_subsurface_interface(SubsurfaceDirection);

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Get surface record of current geometric surface
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;

    template<class T>
    inline CELER_FUNCTION T to_record_index(SurfaceTrackPosition pos) const;
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
    states_.surface[track_id_] = init.surface;
    states_.surface_orientation[track_id_] = init.orientation;
    states_.surface_position[track_id_] = SurfaceTrackPosition{0};
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get geometric surface ID the track is currently on.
 *
 * The ID is invalid if the track is not undergoing a boundary crossing.
 */
CELER_FUNCTION GeometricSurfaceId SurfacePhysicsView::surface() const
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
 * Current position of the track in the sub-surfaces, in track-local
 * coordinates.
 */
CELER_FUNCTION SurfaceTrackPosition& SurfacePhysicsView::subsurface_position()
{
    return states_.surface_position[track_id_];
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
    return this->surface_record().subsurface_materials.size();
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
 * Reset the state of a track.
 *
 * Invalidates the surface ID, indicating the track is no longer undergoing
 * boundary crossing.
 */
CELER_FUNCTION void SurfacePhysicsView::reset() const
{
    states_.surface[track_id_] = {};
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface material ID of the current track position.
 */
CELER_FUNCTION OptMatId SurfacePhysicsView::subsurface_material() const
{
    auto const& surface = this->surface_record();

    SubsurfaceMaterialId subsurface_mat_id{this->subsurface_position().get()};
    if (this->orientation() == SubsurfaceDirection::reverse)
    {
        subsurface_mat_id = SubsurfaceMaterialId{
            surface.subsurface_materials.size() - 1 - subsurface_mat_id.get()};
    }
    CELER_ASSERT(subsurface_mat_id < surface.subsurface_materials.size());

    auto material_record_id = surface.subsurface_materials[subsurface_mat_id];
    CELER_ASSERT(material_record_id < params_.subsurface_materials.size());

    return params_.subsurface_materials[material_record_id];
}

//---------------------------------------------------------------------------//
/*!
 * Get the physics surface ID of the subsurface in the given direction.
 */
CELER_FUNCTION PhysicsSurfaceId
SurfacePhysicsView::subsurface_interface(SubsurfaceDirection d) const
{
    auto const& surface = this->surface_record();

    SubsurfaceInterfaceId subsurface_int_id{this->subsurface_position().get()};
    if (d == SubsurfaceDirection::reverse)
    {
        subsurface_int_id--;
    }
    if (this->orientation() == SubsurfaceDirection::reverse)
    {
        subsurface_int_id = SubsurfaceInterfaceId{
            surface.subsurface_interfaces.size() - 1 - subsurface_int_id.get()};
    }
    CELER_ASSERT(subsurface_int_id < surface.subsurface_interfaces.size());

    auto interface_record_id = surface.subsurface_interfaces[subsurface_int_id];
    CELER_ASSERT(interface_record_id < params_.subsurface_interfaces.size());

    return params_.subsurface_interfaces[interface_record_id];
}

//---------------------------------------------------------------------------//
/*!
 * Cross the subsurface interface in the given direction.
 */
CELER_FUNCTION void
SurfacePhysicsView::cross_subsurface_interface(SubsurfaceDirection d)
{
    CELER_EXPECT(
        (d == SubsurfaceDirection::forward && !this->in_post_volume())
        || (d == SubsurfaceDirection::reverse && !this->in_pre_volume()));
    this->subsurface_position() = this->subsurface_position()
                                  + to_signed_offset(d);
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface record of the current geometric surface.
 */
CELER_FUNCTION SurfaceRecord const& SurfacePhysicsView::surface_record() const
{
    return params_.surfaces[this->surface()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
