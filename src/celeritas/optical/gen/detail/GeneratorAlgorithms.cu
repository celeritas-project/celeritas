//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.cu
//---------------------------------------------------------------------------//
#include "GeneratorAlgorithms.hh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
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
 * The sum is calculated over a fixed upper bound of the buffer size. The
 * total is stored as the generator's number of pending photons, and the
 * increase is counted as queued photons. This does not synchronize.
 */
template<class T>
void inclusive_scan_photons(
    ItemsRef<T, MemSpace::device> const& buffer,
    ItemsRef<size_type, MemSpace::device> const& offsets,
    size_type bound,
    GenCountersObserver<MemSpace::device> counters,
    StreamId stream)
{
    CELER_EXPECT(bound > 0 && bound <= buffer.size());
    CELER_EXPECT(offsets.size() == buffer.size());
    CELER_EXPECT(counters);

    ScopedProfiling profile_this{"prefix-sum-counts"};
    auto input = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        BufferedPhotonCount<T>{buffer.data(), buffer_size_ptr(counters)});
    inclusive_sum_async(input, offsets.data(), bound, stream);

    // The last element of the prefix sum is the total
    update_pending(counters,
                   ObserverPtr<size_type const, MemSpace::device>{
                       offsets.data().get() + bound - 1},
                   stream);
}

//---------------------------------------------------------------------------//
/*!
 * Compact the valid distributions and update the buffer size.
 *
 * The distributions in [0, bound) are stably compacted, and the number of
 * valid distributions is stored as the generator's current buffer size. Valid
 * distributions appended past the previous buffer size are counted as queued.
 * This does not synchronize.
 */
template<class T>
void select_valid(ItemsRef<T, MemSpace::device> const& buffer,
                  size_type bound,
                  GenCountersObserver<MemSpace::device> counters,
                  StreamId stream)
{
    CELER_EXPECT(bound <= buffer.size());
    CELER_EXPECT(counters);

    ScopedProfiling profile_this{"select-valid"};
    DeviceVector<size_type> num_selected(1, stream);
    select_if_async(
        buffer.data(), bound, make_observer(num_selected), IsValid{}, stream);
    set_buffer_size(counters, make_observer(num_selected), stream);
}

//---------------------------------------------------------------------------//
/*!
 * Clear the distributions in [buffer size, bound).
 */
template<class T>
void clear_tail(ItemsRef<T, MemSpace::device> const& buffer,
                size_type bound,
                GenCountersObserver<MemSpace::device> counters,
                StreamId stream)
{
    CELER_EXPECT(bound <= buffer.size());
    CELER_EXPECT(counters);

    if (bound == 0)
    {
        return;
    }
    ClearTailExecutor<T> execute_thread{buffer.data(),
                                        buffer_size_ptr(counters)};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "clear-buffer-tail");
    launch_kernel(bound, stream, execute_thread);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

#define CELER_INST_GENERATOR_ALGORITHMS(T) \
    template void inclusive_scan_photons( \
        ItemsRef<T, MemSpace::device> const&, \
        ItemsRef<size_type, MemSpace::device> const&, \
        size_type, \
        GenCountersObserver<MemSpace::device>, \
        StreamId); \
    template void select_valid(ItemsRef<T, MemSpace::device> const&, \
                               size_type, \
                               GenCountersObserver<MemSpace::device>, \
                               StreamId); \
    template void clear_tail(ItemsRef<T, MemSpace::device> const&, \
                             size_type, \
                             GenCountersObserver<MemSpace::device>, \
                             StreamId)

CELER_INST_GENERATOR_ALGORITHMS(GeneratorDistributionData);
CELER_INST_GENERATOR_ALGORITHMS(WlsDistributionData);

#undef CELER_INST_GENERATOR_ALGORITHMS

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
