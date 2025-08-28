//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeCollectionBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "Types.hh"

namespace celeritas
{
class GeoParamsInterface;
//---------------------------------------------------------------------------//
/*!
 * Create a "collection" that maps from ImplVolumeId to a value.
 *
 * This builds collections for runtime execution using a GeoTrackView's \c
 * impl_volume_id() call, which requires less indirection than \c volume_id() .
 *
 * Given a geometry (which is allowed to be \c GeoParamsInterface ), a function
 * \code T(*)(VolumeId) \endcode will be called for every \c ImplVolumeId in
 * the geometry that corresponds to a canonical volume. The resulting value
 * will be assigned to the collection.
 */
template<class T, class G, class F>
inline Collection<T, Ownership::value, MemSpace::host, ImplVolumeId>
build_volume_collection(G const& geo, F&& fill_value)
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>,
                  "G must be a geometry class");
    static_assert(std::is_convertible_v<std::invoke_result_t<F, VolumeId>, T>,
                  "Incorrect return type for F");

    // Helper lambda to get the appropriate value
    auto fill_or_default = [&](ImplVolumeId iv_id) -> T {
        if (auto vol_id = geo.volume_id(iv_id))
        {
            return fill_value(vol_id);
        }
        return {};
    };

    // Create a collection sized for each implementation volume
    Collection<T, Ownership::value, MemSpace::host, ImplVolumeId> result;
    auto num_impl_volumes = geo.impl_volumes().size();
    resize(&result, num_impl_volumes);

    // Fill values
    for (auto iv_id : range(ImplVolumeId{num_impl_volumes}))
    {
        result[iv_id] = fill_or_default(iv_id);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Helper class to fill a volume collection from a map.
 *
 * Example:
 * \code
   host_data.detector
       = build_volume_collection<DetectorId>(geo, VolumeMapFiller{det_ids});
 * \endcode
 */
template<class T>
class VolumeMapFiller
{
    static_assert(std::is_same_v<typename T::key_type, VolumeId>,
                  "Not a map of volume to value");

  public:
    using mapped_type = typename T::mapped_type;

  public:
    //! Construct with map to use
    explicit VolumeMapFiller(T const& m) : from_vol_(m) {}

    //! Get the value from a volume ID or default if not in the map
    mapped_type operator()(VolumeId vol_id) const
    {
        if (auto iter = from_vol_.find(vol_id); iter != from_vol_.end())
        {
            return iter->second;
        }
        return {};
    }

  private:
    T const& from_vol_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
