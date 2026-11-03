//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.cc
//---------------------------------------------------------------------------//
#include "GeneratorAlgorithms.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/optical/WavelengthShiftData.hh"

#include "../GeneratorData.hh"

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
template<class T>
size_type inclusive_scan_photons(
    ItemsRef<T, MemSpace::host> const& buffer,
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
        acc += GetNumPhotons<T>{}(*data);
        *result++ = acc;
    }

    // Return the final value
    return acc;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * The sum is calculated over a fixed upper bound of the buffer size. The
 * total is stored as the generator's number of pending photons, and the
 * increase is counted as queued photons.
 */
template<class T>
void inclusive_scan_photons(ItemsRef<T, MemSpace::host> const& buffer,
                            ItemsRef<size_type, MemSpace::host> const& offsets,
                            size_type bound,
                            GenCountersObserver<MemSpace::host> counters,
                            StreamId stream)
{
    CELER_EXPECT(bound > 0 && bound <= buffer.size());
    CELER_EXPECT(offsets.size() == buffer.size());
    CELER_EXPECT(counters);

    BufferedPhotonCount<T> get_count{buffer.data(), buffer_size_ptr(counters)};
    auto* result = offsets.data().get();
    size_type acc = 0;
    for (size_type i = 0; i < bound; ++i)
    {
        acc += get_count(i);
        result[i] = acc;
    }

    update_pending(
        counters,
        ObserverPtr<size_type const, MemSpace::host>{result + bound - 1},
        stream);
}

//---------------------------------------------------------------------------//
/*!
 * Compact the valid distributions and update the buffer size.
 *
 * The distributions in [0, bound) are stably compacted, and the number of
 * valid distributions is stored as the generator's current buffer size. Valid
 * distributions appended past the previous buffer size are counted as queued.
 */
template<class T>
void select_valid(ItemsRef<T, MemSpace::host> const& buffer,
                  size_type bound,
                  GenCountersObserver<MemSpace::host> counters,
                  StreamId stream)
{
    CELER_EXPECT(bound <= buffer.size());
    CELER_EXPECT(counters);

    auto* start = buffer.data().get();
    auto* stop = std::remove_if(start, start + bound, LogicalNot{});
    size_type const num_selected = stop - start;
    set_buffer_size(counters,
                    ObserverPtr<size_type const, MemSpace::host>{&num_selected},
                    stream);
}

//---------------------------------------------------------------------------//
/*!
 * Clear the distributions in [buffer size, bound).
 */
template<class T>
void clear_tail(ItemsRef<T, MemSpace::host> const& buffer,
                size_type bound,
                GenCountersObserver<MemSpace::host> counters,
                StreamId)
{
    CELER_EXPECT(bound <= buffer.size());
    CELER_EXPECT(counters);

    launch_kernel(
        bound, ClearTailExecutor<T>{buffer.data(), buffer_size_ptr(counters)});
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::host> const&,
    ItemsRef<size_type, MemSpace::host> const&,
    size_type,
    StreamId);
template size_type inclusive_scan_photons(
    ItemsRef<WlsDistributionData, MemSpace::host> const&,
    ItemsRef<size_type, MemSpace::host> const&,
    size_type,
    StreamId);

#define CELER_INST_GENERATOR_ALGORITHMS(T) \
    template void inclusive_scan_photons( \
        ItemsRef<T, MemSpace::host> const&, \
        ItemsRef<size_type, MemSpace::host> const&, \
        size_type, \
        GenCountersObserver<MemSpace::host>, \
        StreamId); \
    template void select_valid(ItemsRef<T, MemSpace::host> const&, \
                               size_type, \
                               GenCountersObserver<MemSpace::host>, \
                               StreamId); \
    template void clear_tail(ItemsRef<T, MemSpace::host> const&, \
                             size_type, \
                             GenCountersObserver<MemSpace::host>, \
                             StreamId)

CELER_INST_GENERATOR_ALGORITHMS(GeneratorDistributionData);
CELER_INST_GENERATOR_ALGORITHMS(WlsDistributionData);

#undef CELER_INST_GENERATOR_ALGORITHMS

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
