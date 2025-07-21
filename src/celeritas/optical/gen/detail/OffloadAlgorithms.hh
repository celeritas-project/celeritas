//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.hh
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
namespace detail
{
//---------------------------------------------------------------------------//
using celeritas::optical::GeneratorDistributionData;

template<MemSpace M>
using GeneratorDistributionRef
    = Collection<GeneratorDistributionData, Ownership::reference, M>;

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
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
