//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cu
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "TrackInitLauncher.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device.
 */
__global__ void init_tracks_kernel(const ParamsDeviceRef         params,
                                   const StateDeviceRef          states,
                                   const TrackInitStateDeviceRef data)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies = min(data.vacancies.size(), data.initializers.size());

    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < num_vacancies))
        return;

    InitTracksLauncher<MemSpace::device> launch(params, states, data);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the track vector and count secondaries created.
 */
__global__ void locate_alive_kernel(const ParamsDeviceRef         params,
                                    const StateDeviceRef          states,
                                    const TrackInitStateDeviceRef data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    LocateAliveLauncher<MemSpace::device> launch(params, states, data);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
__global__ void process_primaries_kernel(const Span<const Primary> primaries,
                                         const TrackInitStateDeviceRef data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < primaries.size()))
        return;

    ProcessPrimariesLauncher<MemSpace::device> launch(primaries, data);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 */
__global__ void process_secondaries_kernel(const ParamsDeviceRef params,
                                           const StateDeviceRef  states,
                                           const TrackInitStateDeviceRef data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    ProcessSecondariesLauncher<MemSpace::device> launch(params, states, data);
    launch(tid);
}
} // end namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
#define LAUNCH_KERNEL(NAME, THREADS, ARGS...)                                \
    do                                                                       \
    {                                                                        \
        static const KernelParamCalculator NAME##_ckp(NAME##_kernel, #NAME); \
        auto                               kp = NAME##_ckp(THREADS);         \
                                                                             \
        NAME##_kernel<<<kp.grid_size, kp.block_size>>>(ARGS);                \
        CELER_CUDA_CHECK_ERROR();                                            \
    } while (0)

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on device.
 */
void init_tracks(const ParamsDeviceRef&         params,
                 const StateDeviceRef&          states,
                 const TrackInitStateDeviceRef& data)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies = min(data.vacancies.size(), data.initializers.size());

    LAUNCH_KERNEL(init_tracks, num_vacancies, params, states, data);
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the vector of tracks and count secondaries created.
 */
void locate_alive(const ParamsDeviceRef&         params,
                  const StateDeviceRef&          states,
                  const TrackInitStateDeviceRef& data)
{
    LAUNCH_KERNEL(locate_alive, states.size(), params, states, data);
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
void process_primaries(Span<const Primary>            primaries,
                       const TrackInitStateDeviceRef& data)
{
    CELER_EXPECT(primaries.size() <= data.initializers.size());

    LAUNCH_KERNEL(process_primaries, primaries.size(), primaries, data);
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondary particles.
 */
void process_secondaries(const ParamsDeviceRef&         params,
                         const StateDeviceRef&          states,
                         const TrackInitStateDeviceRef& data)
{
    CELER_EXPECT(states.size() <= data.secondary_counts.size());
    CELER_EXPECT(states.size() <= states.interactions.size());

    LAUNCH_KERNEL(process_secondaries, states.size(), params, states, data);
}

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type> vacancies)
{
    thrust::device_ptr<size_type> end = thrust::remove_if(
        thrust::device_pointer_cast(vacancies.data()),
        thrust::device_pointer_cast(vacancies.data() + vacancies.size()),
        IsEqual{flag_id()});

    CELER_CUDA_CHECK_ERROR();

    // New size of the vacancy vector
    size_type result = thrust::raw_pointer_cast(end) - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of surviving secondaries.
 */
template<>
size_type reduce_counts<MemSpace::device>(Span<size_type> counts)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        size_type(0),
        thrust::plus<size_type>());

    CELER_CUDA_CHECK_ERROR();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of surviving secondaries from each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 */
template<>
void exclusive_scan_counts<MemSpace::device>(Span<size_type> counts)
{
    thrust::exclusive_scan(
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + counts.size(),
        counts.data(),
        size_type(0));

    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
#undef LAUNCH_KERNEL
} // namespace detail
} // namespace celeritas
