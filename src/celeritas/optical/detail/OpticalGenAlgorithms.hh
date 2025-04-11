//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

#include "../GeneratorDistributionData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
using GeneratorDistributionRef
    = Collection<::celeritas::optical::GeneratorDistributionData,
                 Ownership::reference,
                 M>;

//---------------------------------------------------------------------------//
struct IsInvalid
{
    // Check if the distribution data is valid
    CELER_FUNCTION bool
    operator()(celeritas::optical::GeneratorDistributionData const& data) const
    {
        return !data;
    }
};

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
