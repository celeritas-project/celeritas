//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
template<MemSpace M>
size_type remove_if_alive(Span<size_type> vacancies);

template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies);
template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type> vacancies);

//---------------------------------------------------------------------------//
// Calculate the exclusive prefix sum of the number of surviving secondaries
template<MemSpace M>
size_type exclusive_scan_counts(Span<size_type> counts);

template<>
size_type exclusive_scan_counts<MemSpace::host>(Span<size_type> counts);
template<>
size_type exclusive_scan_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline size_type remove_if_alive<MemSpace::device>(Span<size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<>
inline size_type exclusive_scan_counts<MemSpace::device>(Span<size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
