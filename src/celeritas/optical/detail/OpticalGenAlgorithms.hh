//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/optical/OpticalDistributionData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct IsInvalid
{
    // Check if the distribution data is valid
    CELER_FUNCTION bool operator()(OpticalDistributionData const& data) const
    {
        return !data;
    }
};

//---------------------------------------------------------------------------//
// Remove all invalid distributions from the buffer.
size_type remove_if_invalid(
    Collection<OpticalDistributionData, Ownership::reference, MemSpace::host> const&,
    size_type,
    size_type,
    StreamId);
size_type remove_if_invalid(Collection<OpticalDistributionData,
                                       Ownership::reference,
                                       MemSpace::device> const&,
                            size_type,
                            size_type,
                            StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline size_type remove_if_invalid(Collection<OpticalDistributionData,
                                              Ownership::reference,
                                              MemSpace::device> const&,
                                   size_type,
                                   size_type,
                                   StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
