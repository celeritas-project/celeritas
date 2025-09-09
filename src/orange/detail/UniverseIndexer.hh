//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"

#include "../OrangeData.hh"
#include "../OrangeTypes.hh"

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
class UniverseIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using UniverseIndexerDataRef
        = UniverseIndexerData<Ownership::const_reference, MemSpace::native>;

    struct LocalSurface
    {
        UniverseId universe;
        LocalSurfaceId surface;
    };

    struct LocalVolume
    {
        UniverseId universe;
        LocalVolumeId volume;
    };
    //!@}

  public:
    // Construct from UniverseIndexerData
    explicit inline CELER_FUNCTION
    UniverseIndexer(UniverseIndexerDataRef const& data);

    // Local-to-global
    inline CELER_FUNCTION ImplSurfaceId
    global_surface(UniverseId uni, LocalSurfaceId surface) const;
    inline CELER_FUNCTION ImplVolumeId global_volume(UniverseId uni,
                                                     LocalVolumeId volume) const;

    // Global-to-local
    inline CELER_FUNCTION LocalSurface local_surface(ImplSurfaceId id) const;
    inline CELER_FUNCTION LocalVolume local_volume(ImplVolumeId id) const;

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
    using DataRef
        = Collection<size_type, Ownership::const_reference, MemSpace::native>;
    using AllVals = AllItems<size_type, MemSpace::native>;
    using SizeId = OpaqueId<size_type>;
    using SpanIter = typename DataRef::SpanConstT::const_iterator;

    //// DATA ////
    UniverseIndexerDataRef const& data_;

    //// IMPLEMENTATION METHODS ////
    static inline CELER_FUNCTION SpanIter find_local(DataRef offsets,
                                                     size_type id);
    static inline CELER_FUNCTION size_type local_size(DataRef offsets,
                                                      UniverseId uni);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from UniverseIndexerData.
 */
CELER_FUNCTION
UniverseIndexer::UniverseIndexer(UniverseIndexerDataRef const& data)
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
CELER_FUNCTION ImplSurfaceId
UniverseIndexer::global_surface(UniverseId uni, LocalSurfaceId surf) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(surf < this->local_size(data_.surfaces, uni));

    return ImplSurfaceId(data_.surfaces[SizeId{uni.unchecked_get()}]
                         + surf.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Transform local to global volume ID.
 */
CELER_FUNCTION ImplVolumeId
UniverseIndexer::global_volume(UniverseId uni, LocalVolumeId volume) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(volume < this->local_size(data_.volumes, uni));

    return ImplVolumeId(data_.volumes[SizeId{uni.unchecked_get()}]
                        + volume.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local surface ID.
 */
CELER_FUNCTION UniverseIndexer::LocalSurface
UniverseIndexer::local_surface(ImplSurfaceId id) const
{
    CELER_EXPECT(id < this->num_surfaces());
    auto iter = this->find_local(data_.surfaces, id.unchecked_get());

    UniverseId uni(iter - data_.surfaces[AllVals{}].begin());
    LocalSurfaceId surface((id - *iter).unchecked_get());
    CELER_ENSURE(uni < this->num_universes());
    return {uni, surface};
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local volume ID.
 */
CELER_FUNCTION UniverseIndexer::LocalVolume
UniverseIndexer::local_volume(ImplVolumeId id) const
{
    CELER_EXPECT(id < this->num_volumes());
    auto iter = this->find_local(data_.volumes, id.unchecked_get());

    UniverseId uni(iter - data_.volumes[AllVals{}].begin());
    LocalVolumeId volume((id - *iter).unchecked_get());
    CELER_ENSURE(uni.get() < this->num_universes());
    return {uni, volume};
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION METHODS
//---------------------------------------------------------------------------//
/*!
 * Locate the given ID in the list of offsets.
 */
CELER_FUNCTION UniverseIndexer::SpanIter
UniverseIndexer::find_local(DataRef offsets, size_type id)
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
CELER_FUNCTION size_type UniverseIndexer::local_size(DataRef offsets,
                                                     UniverseId uni)
{
    CELER_EXPECT(uni && uni.unchecked_get() + 1 < offsets.size());
    return offsets[SizeId{uni.unchecked_get() + 1}]
           - offsets[SizeId{uni.unchecked_get()}];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
