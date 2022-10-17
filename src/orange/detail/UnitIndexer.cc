//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitIndexer.cc
//---------------------------------------------------------------------------//

#include "UnitIndexer.hh"

#include <algorithm>

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Accumulate the elements in the vector, appending the final count.
 */
std::vector<size_type> count_to_offset(std::vector<size_type> v)
{
    // Convert each vector from sizes to accumulated offset
    size_type accum = 0;
    for (size_type& count : v)
    {
        size_type num_local = count;
        count               = accum;
        accum += num_local;
    }
    // Add a final entry: the total count
    v.push_back(accum);

    return v;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * \brief Construct from sizes for each universe ID.
 */
UnitIndexer::UnitIndexer(VecSize num_surfaces, VecSize num_volumes)
{
    CELER_EXPECT(!num_surfaces.empty());
    CELER_EXPECT(num_surfaces.size() == num_volumes.size());

    surfaces_ = count_to_offset(std::move(num_surfaces));
    volumes_  = count_to_offset(std::move(num_volumes));

    CELER_ENSURE(surfaces_.size() >= 2);
    CELER_ENSURE(surfaces_.size() == volumes_.size());
    CELER_ENSURE(this->num_volumes() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Local-to-global
 */
auto UnitIndexer::global_surface(UniverseId uni, SurfaceId surf) const
    -> SurfaceId
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(surf < this->local_size(surfaces_, uni));

    return SurfaceId(surfaces_[uni.unchecked_get()] + surf.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Local-to-global
 */
auto UnitIndexer::global_volume(UniverseId uni, VolumeId volume) const
    -> VolumeId
{
    CELER_EXPECT(uni < this->num_universes());
    CELER_EXPECT(volume < this->local_size(volumes_, uni));

    return VolumeId(volumes_[uni.unchecked_get()] + volume.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Global-to-local
 */
auto UnitIndexer::local_surface(SurfaceId id) const -> LocalSurface
{
    CELER_EXPECT(id < this->num_surfaces());
    auto iter = this->find_local(surfaces_, id.unchecked_get());

    UniverseId uni(iter - surfaces_.begin());
    SurfaceId  surface(id - *iter);
    CELER_ENSURE(uni.get() < this->num_universes());
    return std::make_tuple(uni, surface);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Global-to-local
 */
auto UnitIndexer::local_volume(VolumeId id) const -> LocalVolume
{
    CELER_EXPECT(id < this->num_volumes());
    auto iter = this->find_local(volumes_, id.unchecked_get());

    UniverseId uni(iter - volumes_.begin());
    VolumeId   volume(id - *iter);
    CELER_ENSURE(uni.get() < this->num_universes());
    return std::make_tuple(uni, volume);
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION METHODS
//---------------------------------------------------------------------------//
/*!
 * \brief Locate the given ID in the list of offsets.
 */
auto UnitIndexer::find_local(const VecSize& offsets, size_type id)
    -> VecSize::const_iterator
{
    CELER_EXPECT(id < offsets.back());

    // Use upper bound to skip past universes with zero surfaces.
    auto iter = std::upper_bound(offsets.begin(), offsets.end(), id);
    CELER_ASSERT(iter != offsets.begin());
    --iter;

    CELER_ENSURE(*iter <= id);
    CELER_ENSURE(id < *(iter + 1));
    return iter;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of elements in the given universe.
 */
auto UnitIndexer::local_size(const VecSize& offsets, UniverseId uni)
    -> size_type
{
    CELER_EXPECT(uni && uni.unchecked_get() + 1 < offsets.size());
    return offsets[uni.unchecked_get() + 1] - offsets[uni.unchecked_get()];
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
