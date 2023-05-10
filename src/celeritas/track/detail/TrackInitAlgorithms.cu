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
#include "corecel/data/ObserverPtr.device.hh"

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
size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::device> const&
        vacancies)
{
    auto start = device_pointer_cast(vacancies.data());
    auto end = thrust::remove_if(
        start, start + vacancies.size(), IsEqual{occupied()});
    CELER_DEVICE_CHECK_ERROR();

    // New size of the vacancy vector
    return end - start;
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
size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&
        counts)
{
    // Exclusive scan:
    auto data = device_pointer_cast(counts.data());
    auto stop = thrust::exclusive_scan(
        data, data + counts.size(), data, size_type(0));
    CELER_DEVICE_CHECK_ERROR();

    // Copy the last element (accumulated total) back to host
    return *(stop - 1);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
