//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenAlgorithms.cc
//---------------------------------------------------------------------------//
#include "OpticalGenAlgorithms.hh"

#include <algorithm>
#include <numeric>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct AccumNumPhotons
{
    // Accumulate the number of optical photons from the distribution data
    CELER_FUNCTION size_type
    operator()(size_type count,
               celeritas::optical::GeneratorDistributionData const& data) const
    {
        return count + data.num_photons;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 *
 * This returns the total number of valid distributions in the buffer.
 */
size_type
remove_if_invalid(Collection<celeritas::optical::GeneratorDistributionData,
                             Ownership::reference,
                             MemSpace::host> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId)
{
    auto* start = static_cast<celeritas::optical::GeneratorDistributionData*>(
        buffer.data());
    auto* stop = std::remove_if(start + offset, start + size, IsInvalid{});
    return stop - start;
}

//---------------------------------------------------------------------------//
/*!
 * Count the number of optical photons in the distributions.
 */
size_type
count_num_photons(Collection<celeritas::optical::GeneratorDistributionData,
                             Ownership::reference,
                             MemSpace::host> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId)
{
    auto* start = static_cast<celeritas::optical::GeneratorDistributionData*>(
        buffer.data());
    size_type count = std::accumulate(
        start + offset, start + size, size_type(0), AccumNumPhotons{});
    return count;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
