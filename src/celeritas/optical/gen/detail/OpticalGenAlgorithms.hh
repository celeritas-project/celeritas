//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OpticalGenAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"

#include "../GeneratorDistributionData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
using GeneratorDistributionRef
    = Collection<GeneratorDistributionData, Ownership::reference, M>;

//---------------------------------------------------------------------------//
struct IsInvalid
{
    // Check if the distribution data is valid
    CELER_FUNCTION bool operator()(GeneratorDistributionData const& data) const
    {
        return !data;
    }
};

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
// Remove all invalid distributions from the buffer.
size_type remove_if_invalid(GeneratorDistributionRef<MemSpace::host> const&,
                            size_type,
                            size_type,
                            StreamId);
size_type remove_if_invalid(GeneratorDistributionRef<MemSpace::device> const&,
                            size_type,
                            size_type,
                            StreamId);

//---------------------------------------------------------------------------//
// Count the number of optical photons in the distributions.
size_type count_num_photons(GeneratorDistributionRef<MemSpace::host> const&,
                            size_type,
                            size_type,
                            StreamId);
size_type count_num_photons(GeneratorDistributionRef<MemSpace::device> const&,
                            size_type,
                            size_type,
                            StreamId);

//---------------------------------------------------------------------------//
// Calculate the inclusive prefix sum of the number of optical photons
size_type inclusive_scan_photons(
    GeneratorDistributionRef<MemSpace::host> const&,
    Collection<size_type, Ownership::reference, MemSpace::host> const&,
    size_type,
    StreamId);
size_type inclusive_scan_photons(
    GeneratorDistributionRef<MemSpace::device> const&,
    Collection<size_type, Ownership::reference, MemSpace::device> const&,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline size_type
remove_if_invalid(GeneratorDistributionRef<MemSpace::device> const&,
                  size_type,
                  size_type,
                  StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

inline size_type
count_num_photons(GeneratorDistributionRef<MemSpace::device> const&,
                  size_type,
                  size_type,
                  StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

inline size_type inclusive_scan_photons(
    GeneratorDistributionRef<MemSpace::device> const&,
    Collection<size_type, Ownership::reference, MemSpace::device> const&,
    size_type,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
