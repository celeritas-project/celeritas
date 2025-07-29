//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeSurfaceView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

#include "SurfaceData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access surface properties attached to a volume.
 *
 * This class provides a view into surface data for a specific volume, usually
 * an exiting volume, allowing access to its optional boundary surfaces
 * (surrounding the entire volume) and optional interface surfaces to an
 * adjacent volume.
 */
class VolumeSurfaceView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfaceParamsData>;
    //!@}

  public:
    // Construct from params and pre-step volume ID
    inline CELER_FUNCTION
    VolumeSurfaceView(SurfaceParamsRef const& params, VolumeId id);

    // ID of the Volume
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;

    // ID of the boundary surface for this volume, if any
    CELER_FORCEINLINE_FUNCTION SurfaceId boundary_id() const;

    // Check if this volume has least one interface surface
    CELER_FORCEINLINE_FUNCTION bool has_interface() const;

    // Find surface ID for a transition to another volume instance
    inline CELER_FUNCTION SurfaceId
    find_interface(VolumeInstanceId pre_id, VolumeInstanceId post_id) const;

  private:
    SurfaceParamsRef const& params_;
    VolumeId volume_;

    // HELPER FUNCTIONS

    CELER_FORCEINLINE_FUNCTION VolumeSurfaceRecord const& volume_record() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from surface parameters and volume ID.
 */
CELER_FUNCTION
VolumeSurfaceView::VolumeSurfaceView(SurfaceParamsRef const& params,
                                     VolumeId id)
    : params_(params), volume_(id)
{
    CELER_EXPECT(id < params.volume_surfaces.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID being viewed.
 */
CELER_FUNCTION auto VolumeSurfaceView::volume_id() const -> VolumeId
{
    return volume_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the optional boundary surface ID for this volume.

 * If the result is null, no boundary surface is present.
 */
CELER_FUNCTION SurfaceId VolumeSurfaceView::boundary_id() const
{
    return this->volume_record().boundary;
}

//---------------------------------------------------------------------------//
/*!
 * Check if this volume has at least one interface surface.
 */
CELER_FUNCTION bool VolumeSurfaceView::has_interface() const
{
    return !this->volume_record().surface.empty();
}

//---------------------------------------------------------------------------//
/*!
 * Find the surface ID for a transition between volume instances.
 *
 * This searches for the surface ID associated with a pre->post
 * volume instance transition.
 *
 * \todo The current implementation uses linear search, which is unsuitable for
 * complex detectors such as LHCB's RICH, whose pvRichGrandPMTQuartz has 770
 * specific interfaces. We should either implement an \c equal_range function
 * for searching these sorted arrays, or (better) use a hash lookup for {pre,
 * post} -> surface.
 *
 * \return The surface ID if found, or an invalid ID if not found.
 */
CELER_FUNCTION SurfaceId VolumeSurfaceView::find_interface(
    VolumeInstanceId pre_id, VolumeInstanceId post_id) const
{
    auto const& record = this->volume_record();
    auto get_volinst_id
        = [this](auto item) { return params_.volume_instance_ids[item]; };
    for (size_type index = 0; index < record.interface_pre.size(); ++index)
    {
        {
            VolumeInstanceId cur_pre_id
                = get_volinst_id(record.interface_pre[index]);
            if (pre_id < cur_pre_id)
            {
                // Past range of pre-step IDs: volume isn't in array
                break;
            }
            else if (cur_pre_id < pre_id)
            {
                // Before range; keep going
                continue;
            }
        }
        {
            VolumeInstanceId cur_post_id
                = get_volinst_id(record.interface_post[index]);
            if (post_id < cur_post_id)
            {
                // Past range of post-step IDs
                break;
            }
            else if (cur_post_id < post_id)
            {
                // Before range; keep going
                continue;
            }
        }
        // Pre and post now match
        CELER_ASSERT(index < record.surface.size());
        auto surf_id_offset = record.surface[index];
        CELER_ASSERT(surf_id_offset < params_.surface_ids.size());
        auto result = params_.surface_ids[surf_id_offset];
        CELER_ENSURE(result);
        return result;
    }
    return {};
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Get the volume surface record for the current volume.
 */
CELER_FUNCTION VolumeSurfaceRecord const&
VolumeSurfaceView::volume_record() const
{
    return params_.volume_surfaces[volume_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
