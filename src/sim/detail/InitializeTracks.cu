//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cu
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"
#include "InitTracksLauncher.hh"
#include "LocateAliveLauncher.hh"
#include "ProcessPrimariesLauncher.hh"
#include "ProcessSecondariesLauncher.hh"

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
#define LAUNCH_KERNEL(NAME, THREADS, ...) \
    CELER_LAUNCH_KERNEL(                  \
        NAME, celeritas::device().default_block_size(), THREADS, __VA_ARGS__)

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
} // namespace detail
} // namespace celeritas
