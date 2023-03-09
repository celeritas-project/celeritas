//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.cu
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"

#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
template<>
size_type remove_if_alive<MemSpace::device>(Span<TrackSlotId> vacancies)
{
    thrust::device_ptr<TrackSlotId> end = thrust::remove_if(
        thrust::device_pointer_cast(vacancies.data()),
        thrust::device_pointer_cast(vacancies.data() + vacancies.size()),
        IsEqual{occupied()});

    CELER_DEVICE_CHECK_ERROR();

    // New size of the vacancy vector
    size_type result = thrust::raw_pointer_cast(end) - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of secondaries produced by each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 *
 * The return value is the sum of all elements in the input array.
 */
template<>
size_type exclusive_scan_counts<MemSpace::device>(Span<size_type> counts)
{
    // Copy the last element to the host
    Copier<size_type, MemSpace::device> copy_last_element_to{
        {counts.data() + counts.size() - 1, 1}};
    size_type partial1{};
    copy_last_element_to(MemSpace::host, {&partial1, 1});

    thrust::exclusive_scan(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data() + counts.size()),
        thrust::device_pointer_cast(counts.data()),
        size_type(0));
    CELER_DEVICE_CHECK_ERROR();

    // Copy the last element (the sum of all elements but the last) to the host
    size_type partial2{};
    copy_last_element_to(MemSpace::host, {&partial2, 1});

    return partial1 + partial2;
}

//---------------------------------------------------------------------------//
#undef LAUNCH_KERNEL
}  // namespace detail
}  // namespace celeritas
