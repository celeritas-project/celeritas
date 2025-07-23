//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"

#include "../GeneratorData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
using ItemsRef = Collection<T, Ownership::reference, M>;

//---------------------------------------------------------------------------//
/*!
 * Find the index of the distribution from which to generate the primary.
 *
 * This finds the index in offsets for which offsets[result - 1] <= value <
 * offsets[result].
 */
inline CELER_FUNCTION size_type find_distribution_index(Span<size_type> offsets,
                                                        size_type value)
{
    CELER_EXPECT(!offsets.empty());

    auto iter = celeritas::lower_bound(offsets.begin(), offsets.end(), value);
    CELER_ASSERT(iter != offsets.end());

    if (value == *iter)
    {
        ++iter;
    }
    return iter - offsets.begin();
}

//---------------------------------------------------------------------------//
// Calculate the inclusive prefix sum of the number of optical photons
size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::host> const&,
    ItemsRef<size_type, MemSpace::host> const&,
    size_type,
    StreamId);
size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    ItemsRef<size_type, MemSpace::device> const&,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    ItemsRef<size_type, MemSpace::device> const&,
    size_type,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
