//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.cu
//---------------------------------------------------------------------------//
#include "GeneratorAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_scan.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Thrust.device.hh"
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
 * Whether a buffered distribution is valid.
 */
struct IsValid
{
    template<class T>
    CELER_FUNCTION bool operator()(T const& data) const
    {
        return static_cast<bool>(data);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * \return Total accumulated value
 */
template<class T>
size_type inclusive_scan_photons(
    ItemsRef<T, MemSpace::device> const& buffer,
    ItemsRef<size_type, MemSpace::device> const& offsets,
    size_type size,
    StreamId stream)
{
    CELER_EXPECT(!buffer.empty());
    CELER_EXPECT(size > 0 && size <= buffer.size());
    CELER_EXPECT(offsets.size() == buffer.size());

    ScopedProfiling profile_this{"prefix-sum-counts"};
    auto data = thrust::device_pointer_cast(buffer.data().get());
    auto result = thrust::device_pointer_cast(offsets.data().get());
    auto stop = thrust::transform_inclusive_scan(thrust_execute_on(stream),
                                                 data,
                                                 data + size,
                                                 result,
                                                 GetNumPhotons<T>{},
                                                 thrust::plus<size_type>());
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Copy the last element (accumulated total) back to host
    return load_sync(ObserverPtr<size_type, MemSpace::device>{stop.get() - 1},
                     stream);
}

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

template size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    ItemsRef<size_type, MemSpace::device> const&,
    size_type,
    StreamId);
template size_type inclusive_scan_photons(
    ItemsRef<WlsDistributionData, MemSpace::device> const&,
    ItemsRef<size_type, MemSpace::device> const&,
    size_type,
    StreamId);

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
