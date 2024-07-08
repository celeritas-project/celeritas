//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/OpticalGenAlgorithms.cc
//---------------------------------------------------------------------------//
#include "OpticalGenAlgorithms.hh"

#include <algorithm>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
size_type remove_if_invalid(Collection<OpticalDistributionData,
                                       Ownership::reference,
                                       MemSpace::host> const& buffer,
                            size_type offset,
                            size_type size,
                            StreamId)
{
    auto* start = static_cast<OpticalDistributionData*>(buffer.data());
    auto* stop
        = std::remove_if(start + offset, start + offset + size, IsInvalid{});
    return stop - start;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
