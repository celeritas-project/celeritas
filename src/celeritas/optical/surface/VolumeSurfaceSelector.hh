//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/VolumeSurfaceSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "geocel/SurfaceData.hh"
#include "geocel/VolumeSurfaceView.hh"
#include "celeritas/geo/GeoTrackView.hh"

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
    // Construct with pre-volume IDs
    inline CELER_FUNCTION
    VolumeSurfaceSelector(NativeCRef<SurfaceParamsData> const& params,
                          VolumeId pre_volume,
                          VolumeInstanceId pre_volume_inst);

    // Convenience constructor to use IDs from geometry
    inline CELER_FUNCTION
    VolumeSurfaceSelector(NativeCRef<SurfaceParamsData> const& params,
                          GeoTrackView const& geo);

    // Select surface based on post-volume IDs
    inline CELER_FUNCTION SurfaceId
    operator()(VolumeId post_volume, VolumeInstanceId post_volume_inst) const;

    // Convenience function to use IDs from geometry
    inline CELER_FUNCTION SurfaceId operator()(GeoTrackView const&) const;

  private:
    NativeCRef<SurfaceParamsData> const& params_;
    VolumeId pre_volume_;
    VolumeInstanceId pre_volume_inst_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pre-volume IDs.
 */
CELER_FUNCTION VolumeSurfaceSelector::VolumeSurfaceSelector(
    NativeCRef<SurfaceParamsData> const& params,
    VolumeId pre_volume,
    VolumeInstanceId pre_volume_inst)
    : params_(params)
    , pre_volume_(pre_volume)
    , pre_volume_inst_(pre_volume_inst)
{
    CELER_EXPECT(pre_volume_ < params_.volume_surfaces.size());
    CELER_EXPECT(pre_volume_inst_);
}

//---------------------------------------------------------------------------//
/*!
 * Convenience constructor to use IDs from geometry.
 */
CELER_FUNCTION VolumeSurfaceSelector::VolumeSurfaceSelector(
    NativeCRef<SurfaceParamsData> const& params, GeoTrackView const& geo)
    : VolumeSurfaceSelector(params, geo.volume_id(), geo.volume_instance_id())
{
    CELER_EXPECT(!geo.is_outside());
}

//---------------------------------------------------------------------------//
/*!
 * Select surface based on post-volume IDs.
 *
 * Returns an invalid \c SurfaceId if no surface data exists for the volumes.
 */
CELER_FUNCTION SurfaceId VolumeSurfaceSelector::operator()(
    VolumeId post_volume, VolumeInstanceId post_volume_inst) const
{
    VolumeSurfaceView pre_surface{params_, pre_volume_};

    if (auto surface_id
        = pre_surface.find_interface(pre_volume_inst_, post_volume_inst))
    {
        return surface_id;
    }

    if (auto surface_id = pre_surface.boundary_id())
    {
        return surface_id;
    }

    return VolumeSurfaceView{params_, post_volume}.boundary_id();
}

//---------------------------------------------------------------------------//
/*!
 * Convenience function to use IDs from geometry.
 */
CELER_FUNCTION SurfaceId
VolumeSurfaceSelector::operator()(GeoTrackView const& geo) const
{
    if (geo.is_outside())
        return SurfaceId{};

    return (*this)(geo.volume_id(), geo.volume_instance_id());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
