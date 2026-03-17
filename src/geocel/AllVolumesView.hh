//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/AllVolumesView.hh
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "geocel/Types.hh"

#include "VolumeData.hh"
#include "VolumeView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * GPU-compatible view of the full volume hierarchy graph.
 *
 * This provides device-accessible accessors over the complete
 * \c VolumeParamsData: scalar graph properties (world ID, depth) and
 * instance-to-volume mapping, as well as per-volume \c VolumeView objects.
 */
class AllVolumesView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<VolumeParamsData>;
    //!@}

  public:
    // Construct with shared params data
    explicit inline CELER_FUNCTION AllVolumesView(ParamsRef const& params);

    //! Root volume of the geometry graph
    inline CELER_FUNCTION VolumeId world() const;

    //! Number of logical volumes (nodes)
    inline CELER_FUNCTION VolumeId::size_type num_volumes() const;

    //! Number of volume instances (edges)
    inline CELER_FUNCTION VolumeInstanceId::size_type
    num_volume_instances() const;

    //! Depth of the volume graph (1 for a world with no children)
    inline CELER_FUNCTION VolumeLevelId::size_type num_volume_levels() const;

    // Get the logical volume instantiated by a volume instance
    inline CELER_FUNCTION VolumeId volume_id(VolumeInstanceId vi_id) const;

    // Construct a view for the given volume
    inline CELER_FUNCTION VolumeView volume(VolumeId vol_id) const;

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared params data.
 */
CELER_FORCEINLINE_FUNCTION
AllVolumesView::AllVolumesView(ParamsRef const& params) : params_(params) {}

//---------------------------------------------------------------------------//
/*!
 * Get the root volume of the geometry graph.
 */
CELER_FORCEINLINE_FUNCTION VolumeId AllVolumesView::world() const
{
    return params_.scalars.world;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of logical volumes.
 */
CELER_FORCEINLINE_FUNCTION VolumeId::size_type
AllVolumesView::num_volumes() const
{
    return params_.scalars.num_volumes;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of volume instances.
 */
CELER_FORCEINLINE_FUNCTION VolumeInstanceId::size_type
AllVolumesView::num_volume_instances() const
{
    return params_.scalars.num_volume_instances;
}

//---------------------------------------------------------------------------//
/*!
 * Get the depth of the volume graph.
 */
CELER_FORCEINLINE_FUNCTION VolumeLevelId::size_type
AllVolumesView::num_volume_levels() const
{
    return params_.scalars.num_volume_levels;
}

//---------------------------------------------------------------------------//
/*!
 * Get the logical volume instantiated by a volume instance.
 */
CELER_FUNCTION VolumeId AllVolumesView::volume_id(VolumeInstanceId vi_id) const
{
    CELER_EXPECT(vi_id < params_.volume_ids.size());
    return params_.volume_ids[vi_id];
}

//---------------------------------------------------------------------------//
/*!
 * Construct a \c VolumeView for the given volume.
 */
CELER_FUNCTION VolumeView AllVolumesView::volume(VolumeId vol_id) const
{
    return VolumeView{params_, vol_id};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
