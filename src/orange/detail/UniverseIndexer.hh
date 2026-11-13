//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

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
    using DataRef = NativeCRef<UniverseIndexerData>;

    struct LocalSurface
    {
        UnivId univ;
        LocalSurfaceId surface;
    };

    struct LocalVolume
    {
        UnivId univ;
        LocalVolumeId volume;
    };
    //!@}

  public:
    // Construct from UniverseIndexerData
    explicit inline CELER_FUNCTION UniverseIndexer(DataRef const& data);

    // Local-to-global
    inline CELER_FUNCTION ImplSurfaceId global_surface(
        UnivId uni, LocalSurfaceId surface) const;
    inline CELER_FUNCTION ImplVolumeId global_volume(
        UnivId uni, LocalVolumeId volume) const;

    // Global-to-local
    inline CELER_FUNCTION LocalSurface local_surface(ImplSurfaceId id) const;
    inline CELER_FUNCTION LocalVolume local_volume(ImplVolumeId id) const;

    //! Total number of universes
    CELER_FORCEINLINE_FUNCTION UnivId::size_type num_universes() const
    {
        return data_.surfaces.size() - 1;
    }

    //! Total number of surfaces
    CELER_FUNCTION ImplSurfaceId::size_type num_surfaces() const
    {
        ImplSurfaceId total = data_.surfaces[this->end_uni()];
        return *total;
    }

    //! Number of surfaces in a universe
    CELER_FUNCTION LocalSurfaceId::size_type num_surfaces(UnivId uni) const
    {
        return this->local_size(data_.surfaces, uni);
    }

    //! Total number of volumes
    CELER_FUNCTION ImplVolumeId::size_type num_volumes() const
    {
        ImplVolumeId total = data_.volumes[this->end_uni()];
        return *total;
    }

    //! Number of volumes in a universe
    CELER_FUNCTION LocalVolumeId::size_type num_volumes(UnivId uni) const
    {
        return this->local_size(data_.volumes, uni);
    }

  private:
    //// DATA ////
    DataRef const& data_;

    //// IMPLEMENTATION METHODS ////
    CELER_FORCEINLINE_FUNCTION UnivId end_uni() const
    {
        return UnivId{this->num_universes()};
    }

    template<class IdT, Ownership W, MemSpace M>
    static inline CELER_FUNCTION UnivId find_local(
        Collection<IdT, W, M, UnivId> offsets, IdT id);

    template<class IdT, Ownership W, MemSpace M>
    static inline CELER_FUNCTION typename IdT::size_type local_size(
        Collection<IdT, W, M, UnivId> offsets, UnivId uni);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from UniverseIndexerData.
 *
 * Note that the front values for both surface and volume are zero by
 * construction.
 */
CELER_FUNCTION UniverseIndexer::UniverseIndexer(DataRef const& data)
    : data_(data)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Transform local to global surface ID.
 */
CELER_FUNCTION ImplSurfaceId UniverseIndexer::global_surface(
    UnivId uni, LocalSurfaceId surf) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(surf < this->local_size(data_.surfaces, uni));

    ImplSurfaceId offset = data_.surfaces[uni];
    return offset + *surf;
}

//---------------------------------------------------------------------------//
/*!
 * Transform local to global volume ID.
 */
CELER_FUNCTION ImplVolumeId UniverseIndexer::global_volume(
    UnivId uni, LocalVolumeId volume) const
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(volume < this->local_size(data_.volumes, uni));

    ImplVolumeId offset = data_.volumes[uni];
    return offset + *volume;
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local surface ID.
 */
CELER_FUNCTION auto UniverseIndexer::local_surface(ImplSurfaceId id) const
    -> LocalSurface
{
    CELER_EXPECT(id < this->num_surfaces());

    auto uni = this->find_local(data_.surfaces, id);
    auto surface = id_cast<LocalSurfaceId>(id - data_.surfaces[uni]);
    CELER_ENSURE(uni < this->num_universes());
    return {uni, surface};
}

//---------------------------------------------------------------------------//
/*!
 * Transform global to local volume ID.
 */
CELER_FUNCTION auto UniverseIndexer::local_volume(ImplVolumeId id) const
    -> LocalVolume
{
    CELER_EXPECT(id < this->num_volumes());

    auto uni = this->find_local(data_.volumes, id);
    auto volume = id_cast<LocalVolumeId>(id - data_.volumes[uni]);
    CELER_ENSURE(uni < this->num_universes());
    return {uni, volume};
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION METHODS
//---------------------------------------------------------------------------//
/*!
 * Locate the given ID in the list of offsets.
 */
template<class IdT, Ownership W, MemSpace M>
CELER_FUNCTION UnivId UniverseIndexer::find_local(
    Collection<IdT, W, M, UnivId> offsets, IdT id)
{
    CELER_EXPECT(id < offsets[UnivId{offsets.size() - 1}]);

    // Use upper bound to skip past universes with zero surfaces.
    auto offset_span = offsets[AllItems<IdT>{}];
    auto iter = upper_bound(offset_span.begin(), offset_span.end(), id);

    CELER_ASSERT(iter != offset_span.end());
    --iter;

    IdT offset = *iter;
    IdT next_offset = *(iter + 1);
    CELER_ENSURE(offset <= id);
    CELER_ENSURE(id < next_offset);
    return id_cast<UnivId>(iter - offset_span.begin());
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of elements in the given universe.
 */
template<class IdT, Ownership W, MemSpace M>
CELER_FUNCTION auto UniverseIndexer::local_size(
    Collection<IdT, W, M, UnivId> offsets, UnivId uni) ->
    typename IdT::size_type
{
    CELER_EXPECT(uni && uni + 1 < offsets.size());
    IdT end = offsets[uni + 1];
    IdT begin = offsets[uni];
    return *end - *begin;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
