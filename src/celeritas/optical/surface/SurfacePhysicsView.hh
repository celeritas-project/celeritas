//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/Types.hh"

#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Persistent optical surface physics data for a track.
 *
 * Maps surface track positions to material and interface data for an optical
 * surface. Some persistent track data (orientation, pre-material, and
 * post-material) are track-dependent and used for the mapping.
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    //!@}

  public:
    // Construct view from data and state
    inline CELER_FUNCTION
    SurfacePhysicsView(SurfaceParamsRef const&, SurfaceId, SubsurfaceDirection);

    // Get current surface ID
    inline CELER_FUNCTION SurfaceId surface() const;

    // Get surface orientation
    inline CELER_FUNCTION SubsurfaceDirection orientation() const;

    // Get optical material at the given track position
    inline CELER_FUNCTION OptMatId material(SurfaceTrackPosition) const;

    // Get the physics surface at the given position and direction
    inline CELER_FUNCTION PhysSurfaceId interface(SurfaceTrackPosition,
                                                  SubsurfaceDirection) const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceId surface_;
    SubsurfaceDirection orientation_;

    // Get record data for the current surface
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data, states, and a given track ID.
 */
CELER_FUNCTION
SurfacePhysicsView::SurfacePhysicsView(SurfaceParamsRef const& params,
                                       SurfaceId surface,
                                       SubsurfaceDirection orientation)
    : params_(params), surface_(surface), orientation_(orientation)
{
    CELER_EXPECT(surface_ < params_.surfaces.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get geometric surface ID the track is currently on.
 *
 * The ID is invalid if the track is not undergoing a boundary crossing.
 */
CELER_FUNCTION SurfaceId SurfacePhysicsView::surface() const
{
    return surface_;
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
    return orientation_;
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface material ID of the given track position.
 */
CELER_FUNCTION OptMatId SurfacePhysicsView::material(SurfaceTrackPosition pos) const
{
    CELER_ASSERT(pos < this->surface_record().subsurface_materials.size());

    auto material_record_id = OrientedItemMap{
        this->surface_record().subsurface_materials, this->orientation()}[pos];
    CELER_ASSERT(material_record_id < params_.subsurface_materials.size());

    return params_.subsurface_materials[material_record_id];
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface interface ID of the given track position and
 * direction.
 */
CELER_FUNCTION PhysSurfaceId SurfacePhysicsView::interface(
    SurfaceTrackPosition pos, SubsurfaceDirection d) const
{
    auto interface_pos = pos + IfReverseDirection<int>{-1}(d);

    CELER_ASSERT(interface_pos
                 < this->surface_record().subsurface_interfaces.size());

    return OrientedItemMap{this->surface_record().subsurface_interfaces,
                           this->orientation()}[interface_pos];
}

//---------------------------------------------------------------------------//
/*!
 * Get surface record of current geometric surface.
 */
CELER_FUNCTION SurfaceRecord const& SurfacePhysicsView::surface_record() const
{
    return params_.surfaces[this->surface()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
