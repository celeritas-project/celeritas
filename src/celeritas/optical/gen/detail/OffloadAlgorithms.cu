//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "corecel/Assert.hh"
#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/ThreadId.hh"

#include "GeneratorAlgorithms.hh"
#include "GeneratorCounterAlgorithms.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Copy the selected distributions to the end of the buffer
struct AppendExecutor
{
    ObserverPtr<GeneratorDistributionData> data;
    ObserverPtr<GeneratorDistributionData const> selected;
    ObserverPtr<size_type const> num_selected;
    ObserverPtr<size_type const> buffer_size;

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        if (tid.get() < *num_selected)
        {
            data[*buffer_size + tid.get()] = selected[tid.get()];
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Append the valid distributions created in this step to the buffer.
 *
 * The distributions written by each track slot past the buffered data are
 * stably compacted, the buffer size is increased by the number of valid
 * distributions, and their photons are added to the number of pending tracks.
 * Since the buffer size is read on device, this does not synchronize.
 */
void append_distributions(
    optical::CoreState<MemSpace::device>& state,
    ItemsRef<GeneratorDistributionData, MemSpace::device> const& buffer,
    ObserverPtr<GeneratorStateCounters, MemSpace::device> counters,
    size_type num_slots,
    StreamId stream_id)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(num_slots <= buffer.size());

    ScopedProfiling profile_this{"append-distributions"};

    auto buffer_size = optical::detail::buffer_size_ptr(counters);
    auto data = buffer.data();

    // Temporary storage for the compacted distributions, the number of
    // selected distributions, and the number of photons
    DeviceVector<GeneratorDistributionData> selected_vec(num_slots, stream_id);
    DeviceVector<size_type> num_selected_vec(1, stream_id);
    DeviceVector<size_type> num_photons_vec(1, stream_id);
    auto selected = make_observer(selected_vec);
    auto num_selected = make_observer(num_selected_vec);
    auto num_photons = make_observer(num_photons_vec);

    // Compact the distributions created in this step
    auto counting = thrust::make_counting_iterator<size_type>(0);
    copy_if_async(thrust::make_transform_iterator(
                      counting, OffloadedDistribution{data, buffer_size}),
                  selected,
                  num_slots,
                  num_selected,
                  optical::detail::IsValid{},
                  stream_id);

    // Count the photons to generate and add them to the pending tracks
    using PhotonCount
        = optical::detail::BufferedPhotonCount<GeneratorDistributionData>;
    reduce_sum_async(thrust::make_transform_iterator(
                         counting, PhotonCount{selected, num_selected}),
                     num_photons,
                     num_slots,
                     stream_id);
    state.add_pending(num_photons);

    // Append the compacted distributions and update the buffer size
    {
        AppendExecutor execute_thread{
            data, selected, num_selected, buffer_size};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "append-distributions");
        launch_kernel(num_slots, stream_id, execute_thread);
    }
    optical::detail::add_buffer_size(counters, num_selected, stream_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
