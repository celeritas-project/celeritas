//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cc
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <algorithm>
#include <numeric>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"

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
    auto* stop = std::remove_if(start + offset, start + size, LogicalFalse{});
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
}  // namespace detail
}  // namespace celeritas
