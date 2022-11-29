//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/Collection.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit input to params data.
 *
 * Linearize the data in a UnitInput and add it to the host.
 */
class UnitIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using UnitIndexerDataRef
        = UnitIndexerData<Ownership::const_reference, MemSpace::native>;
    using DataRef
        = Collection<size_type, Ownership::const_reference, MemSpace::native>;
    using AllVals  = AllItems<size_type, MemSpace::native>;
    using SizeId   = OpaqueId<size_type>;
    using SpanIter = Span<const size_type>::const_iterator;

    struct LocalSurface
    {
        UniverseId universe;
        SurfaceId  surface;
    };

    struct LocalVolume
    {
        UniverseId universe;
        VolumeId   volume;
    };
    //!@}

  public:
    // Construct from UnitIndexerData
    explicit inline CELER_FUNCTION UnitIndexer(const UnitIndexerDataRef& data);

    // Local-to-global
    CELER_FUNCTION inline SurfaceId
    global_surface(UniverseId uni, SurfaceId surface) const;
    CELER_FUNCTION inline VolumeId
    global_volume(UniverseId uni, VolumeId volume) const;

    // Global-to-local
    CELER_FUNCTION inline LocalSurface local_surface(SurfaceId id) const;
    CELER_FUNCTION inline LocalVolume  local_volume(VolumeId id) const;

    //! Total number of universes
    CELER_FUNCTION size_type num_universes() const
    {
        return data_.surfaces.size() - 1;
    }

    //! Total number of surfaces
    CELER_FUNCTION size_type num_surfaces() const
    {
        return data_.surfaces[AllVals{}].back();
    }

    //! Total number of cells
    CELER_FUNCTION size_type num_volumes() const
    {
        return data_.volumes[AllVals{}].back();
    }

  private:
    //// DATA ////
    UnitIndexerDataRef data_;

    //// IMPLEMENTATION METHODS ////
    static inline CELER_FUNCTION SpanIter  find_local(DataRef   offsets,
                                                      size_type id);
    static inline CELER_FUNCTION size_type local_size(DataRef    offsets,
                                                      UniverseId uni);
    static inline CELER_FUNCTION SpanIter  upper_bound(SpanIter  begin,
                                                       SpanIter  end,
                                                       size_type id);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from UnitIndexerData
 */
CELER_FUNCTION UnitIndexer::UnitIndexer(const UnitIndexerDataRef& data)
    : data_(data)
{
    CELER_EXPECT(data_.surfaces.size() == data_.volumes.size());
    CELER_EXPECT(data_.surfaces[AllVals{}].front() == 0);
    CELER_EXPECT(data_.volumes[AllVals{}].front() == 0);
}

//---------------------------------------------------------------------------//
/*!
 * Transform local to global surface ID.
 */
CELER_FUNCTION SurfaceId UnitIndexer::global_surface(UniverseId uni,
                                                     SurfaceId  surf) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(surf < this->local_size(data_.surfaces, uni));

    return SurfaceId(data_.surfaces[SizeId{uni.unchecked_get()}]
                     + surf.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Transform local to global volume ID.
 */
CELER_FUNCTION VolumeId UnitIndexer::global_volume(UniverseId uni,
                                                   VolumeId   volume) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(volume < this->local_size(data_.volumes, uni));

    return VolumeId(data_.volumes[SizeId{uni.unchecked_get()}]
                    + volume.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local surface ID.
 */
CELER_FUNCTION UnitIndexer::LocalSurface
               UnitIndexer::local_surface(SurfaceId id) const
{
    CELER_EXPECT(id < this->num_surfaces());
    auto iter = this->find_local(data_.surfaces, id.unchecked_get());

    UniverseId uni(iter - data_.surfaces[AllVals{}].begin());
    SurfaceId  surface(id - *iter);
    CELER_ENSURE(uni < this->num_universes());
    return {uni, surface};
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local volume ID.
 */
CELER_FUNCTION UnitIndexer::LocalVolume
               UnitIndexer::local_volume(VolumeId id) const
{
    CELER_EXPECT(id < this->num_volumes());
    auto iter = this->find_local(data_.volumes, id.unchecked_get());

    UniverseId uni(iter - data_.volumes[AllVals{}].begin());
    VolumeId   volume(id - *iter);
    CELER_ENSURE(uni.get() < this->num_universes());
    return {uni, volume};
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION METHODS
//---------------------------------------------------------------------------//
/*!
 * Locate the given ID in the list of offsets.
 */
CELER_FUNCTION UnitIndexer::SpanIter
               UnitIndexer::find_local(DataRef offsets, size_type id)
{
    CELER_EXPECT(id < offsets[SizeId{offsets.size() - 1}]);

    // Use upper bound to skip past universes with zero surfaces.
    auto iter = upper_bound(
        offsets[AllVals{}].begin(), offsets[AllVals{}].end(), id);

    CELER_ASSERT(iter != offsets[AllVals{}].end());
    --iter;

    CELER_ENSURE(*iter <= id);
    CELER_ENSURE(id < *(iter + 1));
    return iter;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of elements in the given universe.
 */
CELER_FUNCTION size_type UnitIndexer::local_size(DataRef    offsets,
                                                 UniverseId uni)
{
    CELER_EXPECT(uni && uni.unchecked_get() + 1 < offsets.size());
    return offsets[SizeId{uni.unchecked_get() + 1}]
           - offsets[SizeId{uni.unchecked_get()}];
}

//---------------------------------------------------------------------------//
/*!
 * Host/device implementaiton of std::upper_bound
 */
CELER_FUNCTION UnitIndexer::SpanIter
               UnitIndexer::upper_bound(UnitIndexer::SpanIter begin,
                         UnitIndexer::SpanIter end,
                         size_type             id)
{
    CELER_EXPECT(begin <= end);

    SpanIter iter = begin;

    while (id >= *iter)
    {
        iter++;
    }

    return iter;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
