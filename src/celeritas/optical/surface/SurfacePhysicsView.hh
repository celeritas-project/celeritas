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
 * Optical surface physics data.
 *
 * Maps surface track position to interstitial optical material and interface
 * data for a given optical surface. The optical surface may be oriented
 * (forward or reverse) relative to its layout in the data record.
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

    // Get optical interstitial material at the given track position
    inline CELER_FUNCTION
        OptMatId interstitial_material(SurfaceTrackPosition) const;

    // Get the physics surface at the given position and direction
    inline CELER_FUNCTION PhysSurfaceId interface(SurfaceTrackPosition,
                                                  SubsurfaceDirection) const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceId surface_;
    SubsurfaceDirection orientation_;

    // Get record data for the current surface
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;

    // Index a map along a given orientation
    template<class T>
    inline CELER_FUNCTION T oriented_map(ItemMap<SurfaceTrackPosition, T> const&,
                                         SurfaceTrackPosition) const;
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
 * Return the interstitial material ID of the given track position.
 *
 * Position should be in the range [1,N] where N is the number of subsurface
 * materials.
 */
CELER_FUNCTION OptMatId
SurfacePhysicsView::interstitial_material(SurfaceTrackPosition pos) const
{
    // Position 0 is pre-volume material, and N+1 is post-volume material.
    // Offset position so it maps to the interstitial material range.
    --pos;
    CELER_ASSERT(pos < this->surface_record().subsurface_materials.size());

    auto material_record_id
        = this->oriented_map(this->surface_record().subsurface_materials, pos);
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
    auto interface_pos = pos + (d == SubsurfaceDirection::reverse ? -1 : 0);

    CELER_ASSERT(interface_pos
                 < this->surface_record().subsurface_interfaces.size());

    return oriented_map(this->surface_record().subsurface_interfaces,
                        interface_pos);
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
/*!
 * Index an \c ItemMap along the surface's orientation.
 */
template<class T>
CELER_FUNCTION T SurfacePhysicsView::oriented_map(
    ItemMap<SurfaceTrackPosition, T> const& map, SurfaceTrackPosition pos) const
{
    auto index = orientation_ == SubsurfaceDirection::reverse
                     ? SurfaceTrackPosition{(map.size() - 1) - pos.get()}
                     : pos;
    CELER_ASSERT(index < map.size());
    return map[index];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
