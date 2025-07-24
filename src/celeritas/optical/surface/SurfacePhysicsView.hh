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
        SubsurfaceDirection orientation;
    };

  public:
    // Construct from params, state, and surface ID for a given track
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             TrackSlotId);

    // Initialize the boundary crossing for the track
    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    // Reset state after leaving surface
    inline CELER_FUNCTION void reset();

    // Get surface ID
    inline CELER_FUNCTION SurfaceId surface_id() const;

    // In the pre-volume subsurface material
    inline CELER_FUNCTION bool in_pre_volume() const;

    // In the post-volume subsurface material
    inline CELER_FUNCTION bool in_post_volume() const;

    // Get current subsurface material
    inline CELER_FUNCTION SubsurfaceMaterialId subsurface_material() const;

    // Number of subsurface materials that make up the surface
    inline CELER_FUNCTION SubsurfaceMaterialId::size_type
    num_subsurface_materials() const;

    // Number of subsurface interfaces
    inline CELER_FUNCTION SubsurfaceInterfaceId::size_type
    num_subsurface_interfaces() const;

    // Subsurface material record ID
    inline CELER_FUNCTION SubsurfaceMaterialRecordId
    subsurface_material_record() const;

    // Subsurface interface record ID in the given direction
    inline CELER_FUNCTION SubsurfaceInterfaceRecordId
    subsurface_interface_record(SubsurfaceDirection d) const;

    // Move across a subsurface interface
    inline CELER_FUNCTION void
    cross_subsurface_interface(SubsurfaceDirection d);

  private:
    SurfaceParamsRef const& params_;
    SurfaceStateRef const& states_;
    TrackSlotId const track_id_;

    // Access track's surface record
    inline CELER_FUNCTION SurfacePhysicsRecord const& surface() const;
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
    states_.surface[track_id_] = init.surface;
    states_.subsurface_material[track_id_] = SubsurfaceMaterialId{0};
    states_.surface_orientation[track_id_] = init.orientation;
    return *this;
}
//---------------------------------------------------------------------------//
/*!
 * Reset the state after leaving the surface.
 */
CELER_FUNCTION void SurfacePhysicsView::reset()
{
    states_.surface[track_id_] = SurfaceId{};
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
 * Whether the current subsurface material is the pre-volume.
 */
CELER_FUNCTION bool SurfacePhysicsView::in_pre_volume() const
{
    return this->subsurface_material() == SubsurfaceMaterialId{0};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the current subsurface material is the post-volume.
 */
CELER_FUNCTION bool SurfacePhysicsView::in_post_volume() const
{
    return this->subsurface_material().get()
           == this->num_subsurface_interfaces();
}

//---------------------------------------------------------------------------//
/*!
 * Get which subsurface material the track is currently on.
 */
CELER_FUNCTION SubsurfaceMaterialId SurfacePhysicsView::subsurface_material() const
{
    return states_.subsurface_material[track_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Number of subsurface materials that make up the surface.
 */
CELER_FUNCTION SubsurfaceMaterialId::size_type
SurfacePhysicsView::num_subsurface_materials() const
{
    return this->surface().subsurface_materials.size();
}

//---------------------------------------------------------------------------//
/*!
 * Number of subsurface interfaces.
 *
 * This should always be one less than the number of subsurface materials,
 * by construction of \c SurfacePhysicsParams .
 */
CELER_FUNCTION SubsurfaceInterfaceId::size_type
SurfacePhysicsView::num_subsurface_interfaces() const
{
    return this->surface().subsurface_interfaces.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the subsurface material record ID.
 */
CELER_FUNCTION SubsurfaceMaterialRecordId
SurfacePhysicsView::subsurface_material_record() const
{
    // Index in track local frame
    SubsurfaceMaterialRecordId index{this->subsurface_material().get()};

    // Flip interval if orientation reversed
    if (states_.surface_orientation[track_id_] == SubsurfaceDirection::reverse)
    {
        index = SubsurfaceMaterialRecordId{this->num_subsurface_materials() - 1
                                           - index.get()};
    }

    return index;
}

//---------------------------------------------------------------------------//
/*!
 * Get the next subsurface interface in the given direction.
 *
 * Returns an out of range ID if directed off of the surface.
 */
CELER_FUNCTION SubsurfaceInterfaceRecordId
SurfacePhysicsView::subsurface_interface_record(SubsurfaceDirection d) const
{
    // Index in track local frame
    SubsurfaceInterfaceRecordId index{this->subsurface_material().get()};
    if (d == SubsurfaceDirection::reverse)
    {
        index--;
    }

    // Flip interval if orientation reversed
    if (states_.surface_orientation[track_id_] == SubsurfaceDirection::reverse)
    {
        index = SubsurfaceInterfaceRecordId{this->num_subsurface_interfaces()
                                            - 1 - index.get()};
    }

    return index;
}

//---------------------------------------------------------------------------//
/*!
 * Move track across a subsurface interface in the given direction.
 */
CELER_FUNCTION void
SurfacePhysicsView::cross_subsurface_interface(SubsurfaceDirection d)
{
    CELER_EXPECT(
        (d == SubsurfaceDirection::forward && !this->in_post_volume())
        || (d == SubsurfaceDirection::reverse && !this->in_pre_volume()));
    states_.subsurface_material[track_id_] = this->subsurface_material()
                                             + static_cast<int>(d);
}

//---------------------------------------------------------------------------//
/*!
 * Access track's surface record.
 */
CELER_FUNCTION SurfacePhysicsRecord const& SurfacePhysicsView::surface() const
{
    CELER_EXPECT(this->surface_id() < params_.surfaces.size());
    return params_.surfaces[this->surface_id()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
