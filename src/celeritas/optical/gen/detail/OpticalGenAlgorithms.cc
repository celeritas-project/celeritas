//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OpticalGenAlgorithms.cc
//---------------------------------------------------------------------------//
#include "OpticalGenAlgorithms.hh"

#include <algorithm>
#include <numeric>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//

struct AccumNumPhotons
{
    // Accumulate the number of optical photons from the distribution data
    size_type
    operator()(size_type count, GeneratorDistributionData const& data) const
    {
        return count + data.num_photons;
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 *
 * \return Total number of valid distributions in the buffer
 */
size_type
remove_if_invalid(GeneratorDistributionRef<MemSpace::host> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId)
{
    auto* start = buffer.data().get();
    auto* stop = std::remove_if(start + offset, start + size, IsInvalid{});
    return stop - start;
}

//---------------------------------------------------------------------------//
/*!
 * Count the number of optical photons in the distributions.
 */
size_type
count_num_photons(GeneratorDistributionRef<MemSpace::host> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId)
{
    auto* start = buffer.data().get();
    size_type count = std::accumulate(
        start + offset, start + size, size_type(0), AccumNumPhotons{});
    return count;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * \return Total accumulated value
 */
size_type inclusive_scan_photons(
    GeneratorDistributionRef<MemSpace::host> const& buffer,
    Collection<size_type, Ownership::reference, MemSpace::host> const& offsets,
    size_type size,
    StreamId)
{
    CELER_EXPECT(!buffer.empty());
    CELER_EXPECT(size > 0 && size <= buffer.size());
    CELER_EXPECT(offsets.size() == buffer.size());

    auto* data = buffer.data().get();
    auto* result = offsets.data().get();

    size_type acc = 0;
    auto* const stop = data + size;
    for (; data != stop; ++data)
    {
        acc += data->num_photons;
        *result++ = acc;
    }

    // Return the final value
    return acc;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
