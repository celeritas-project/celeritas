//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.cc
//---------------------------------------------------------------------------//
#include "GeneratorAlgorithms.hh"

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * \return Total accumulated value
 */
size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::host> const& buffer,
    ItemsRef<size_type, MemSpace::host> const& offsets,
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
}  // namespace optical
}  // namespace celeritas
