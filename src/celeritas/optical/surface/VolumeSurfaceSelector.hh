//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/VolumeSurfaceSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "geocel/VolumeSurfaceView.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Retrieve the surface ID between two volume instances.
 *
 * Given (old, new) physical volumes P0, P1 corresponding to logical volumes
 * L0, L1
 * - Ordered (P0, P1) interface surface
 * - Boundary surface of L0
 * - Boundary surface of L1
 *
 * This behavior differs from Geant4's order of precedence, which considers
 * if there's a mother-daughter relation between L0 and L1 when both have
 * a boundary surface:
 * - Ordered (P0, P1) interface surface
 * - Boundary surface of L1 if it's the daughter of L0
 * - Boundary surface of L0
 * - Boundary surface of L1
 *
 * When multiple layers are implemented, this selector will be responsible
 * for determining the ordering of the layers between the volumes,
 * including both interface and boundary surfaces.
 */
class VolumeSurfaceSelector
{
  public:
    struct OrientedSurface
    {
        SurfaceId surface{};
        SubsurfaceDirection orientation;

        explicit CELER_FUNCTION operator bool() const
        {
            return static_cast<bool>(surface);
        }
    };

  public:
    // Construct with pre-volume IDs
    inline CELER_FUNCTION
    VolumeSurfaceSelector(VolumeSurfaceView pre_surface,
                          VolumeInstanceId pre_volume_inst);

    // Select surface based on post-volume IDs
    inline CELER_FUNCTION OrientedSurface
    operator()(VolumeSurfaceView const& post_volume,
               VolumeInstanceId post_volume_inst) const;

  private:
    VolumeSurfaceView pre_surface_;
    VolumeInstanceId pre_volume_inst_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pre-volume IDs.
 */
CELER_FUNCTION
VolumeSurfaceSelector::VolumeSurfaceSelector(VolumeSurfaceView pre_surface,
                                             VolumeInstanceId pre_volume_inst)
    : pre_surface_(std::move(pre_surface)), pre_volume_inst_(pre_volume_inst)
{
    CELER_EXPECT(pre_volume_inst_);
}

//---------------------------------------------------------------------------//
/*!
 * Select surface based on post-volume IDs.
 *
 * Returns an invalid \c SurfaceId if no surface data exists for the volumes.
 */
CELER_FUNCTION auto
VolumeSurfaceSelector::operator()(VolumeSurfaceView const& post_surface,
                                  VolumeInstanceId post_volume_inst) const
    -> OrientedSurface
{
    // P0 -> P1 interface surface in forward direction
    if (auto surface_id
        = pre_surface_.find_interface(pre_volume_inst_, post_volume_inst))
    {
        return {surface_id, SubsurfaceDirection::forward};
    }

    // L0 boundary surface in forward direction
    if (auto surface_id = pre_surface_.boundary_id())
    {
        return {surface_id, SubsurfaceDirection::forward};
    }

    // Return the L1 boundary surface from the opposite direction.
    // If no boundary surface exists, an invalid OrientedSurface is returned.
    return {post_surface.boundary_id(), SubsurfaceDirection::reverse};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
