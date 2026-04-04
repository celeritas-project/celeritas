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
template<class T>
size_type remove_if_invalid(ItemsRef<T, MemSpace::host> const& buffer,
                            size_type offset,
                            size_type size,
                            StreamId)
{
    auto* start = buffer.data().get();
    auto* stop = std::remove_if(start + offset, start + size, LogicalNot{});
    return stop - start;
}

//---------------------------------------------------------------------------//
/*!
 * Count the number of optical photons in the distributions.
 */
size_type count_num_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::host> const& buffer,
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
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template size_type
remove_if_invalid(ItemsRef<GeneratorDistributionData, MemSpace::host> const&,
                  size_type,
                  size_type,
                  StreamId);
template size_type
remove_if_invalid(ItemsRef<WlsDistributionData, MemSpace::host> const&,
                  size_type,
                  size_type,
                  StreamId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
