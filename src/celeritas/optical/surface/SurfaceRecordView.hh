//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceRecordView.hh
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
 */
class SurfaceRecordView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    //!@}

  public:
    // Create view from surface physics data and state
    inline CELER_FUNCTION
    SurfaceRecordView(SurfaceParamsRef const&, SurfaceId, SubsurfaceDirection);

    inline CELER_FUNCTION OptMatId material(SurfaceTrackPosition) const;
    inline CELER_FUNCTION PhysSurfaceId interface(SurfaceTrackPosition,
                                                  SubsurfaceDirection) const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceId surface_;
    SubsurfaceDirection orientation_;

    // Get surface record of current geometric surface
    inline CELER_FUNCTION SurfaceRecord const& surface_record() const;

    // Get the record index from the track-local position
    template<class T, class U>
    CELER_FUNCTION U to_record_index(SurfaceTrackPosition,
                                     ItemMap<T, U> const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION
SurfaceRecordView::SurfaceRecordView(SurfaceParamsRef const& params,
                                     SurfaceId surface,
                                     SubsurfaceDirection orientation)
    : params_(params), surface_(surface), orientation_(orientation)
{
    CELER_EXPECT(surface_ < params_.surfaces.size());
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface material ID of the given track position.
 */
CELER_FUNCTION OptMatId SurfaceRecordView::material(SurfaceTrackPosition pos) const
{
    CELER_EXPECT(pos.get() > 0);
    CELER_EXPECT(pos - 1 < this->surface_record().subsurface_materials.size());

    auto material_record_id = this->to_record_index(
        pos - 1, this->surface_record().subsurface_materials);

    CELER_ASSERT(material_record_id < params_.subsurface_materials.size());

    return params_.subsurface_materials[material_record_id];
}

//---------------------------------------------------------------------------//
/*!
 * Return the subsurface interface ID of the given track position and
 * direction.
 */
CELER_FUNCTION PhysSurfaceId SurfaceRecordView::interface(
    SurfaceTrackPosition pos, SubsurfaceDirection d) const
{
    auto interface_pos = pos + (static_cast<int>(d) - 1);

    CELER_ASSERT(interface_pos
                 < this->surface_record().subsurface_interfaces.size());

    return this->to_record_index(interface_pos,
                                 this->surface_record().subsurface_interfaces);
}

//---------------------------------------------------------------------------//
/*!
 * Get surface record of current geometric surface.
 */
CELER_FUNCTION SurfaceRecord const& SurfaceRecordView::surface_record() const
{
    return params_.surfaces[surface_];
}

//---------------------------------------------------------------------------//
/*!
 * Convert track-loacl position to index in a surface record.
 */
template<class T, class U>
CELER_FUNCTION U SurfaceRecordView::to_record_index(
    SurfaceTrackPosition pos, ItemMap<T, U> const& map) const
{
    T index{pos.get()};
    if (orientation_ == SubsurfaceDirection::reverse)
    {
        index = T{map.size() - 1 - index.get()};
    }
    CELER_ASSERT(index < map.size());
    return map[index];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
