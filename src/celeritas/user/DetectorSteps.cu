//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cu
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "StepData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Gather results from active tracks that are in a detector.
 */
__global__ void
gather_step_kernel(DeviceRef<StepStateData> const state, size_type num_valid)
{
    CELER_EXPECT(state.size() == state.scratch.size()
                 && state.size() >= num_valid);

    TrackSlotId tid{KernelParamCalculator::thread_id().get()};
    if (!(tid < num_valid))
    {
        return;
    }

#define DS_FAST_GET(CONT, TID) CONT.data().get()[TID.unchecked_get()]

    TrackSlotId valid_tid{DS_FAST_GET(state.valid_id, tid)};
    CELER_ASSERT(valid_tid < state.size());

    // Equivalent to `CONT[tid]` but without debug checking, which causes this
    // function to grow large enough to emit warnings
#define DS_COPY_IF_SELECTED(FIELD)                          \
    do                                                      \
    {                                                       \
        if (!state.data.FIELD.empty())                      \
        {                                                   \
            DS_FAST_GET(state.scratch.FIELD, tid)           \
                = DS_FAST_GET(state.data.FIELD, valid_tid); \
        }                                                   \
    } while (0)

    DS_COPY_IF_SELECTED(detector);
    DS_COPY_IF_SELECTED(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_COPY_IF_SELECTED(points[sp].time);
        DS_COPY_IF_SELECTED(points[sp].pos);
        DS_COPY_IF_SELECTED(points[sp].dir);
        DS_COPY_IF_SELECTED(points[sp].energy);
    }

    DS_COPY_IF_SELECTED(event_id);
    DS_COPY_IF_SELECTED(parent_id);
    DS_COPY_IF_SELECTED(track_step_count);
    DS_COPY_IF_SELECTED(step_length);
    DS_COPY_IF_SELECTED(particle);
    DS_COPY_IF_SELECTED(energy_deposition);
#undef DS_COPY_IF_SELECTED
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Gather results from active tracks that are in a detector.
 */
void gather_step(DeviceRef<StepStateData> const& state, size_type num_valid)
{
    if (num_valid == 0)
    {
        // No valid tracks
        return;
    }

    CELER_LAUNCH_KERNEL(gather_step,
                        num_valid,
                        celeritas::device().stream(state.stream_id).get(),
                        state,
                        num_valid);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
struct HasDetector
{
    CELER_FORCEINLINE_FUNCTION bool operator()(DetectorId const& d)
    {
        return static_cast<bool>(d);
    }
};

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::vector<T>* dst,
                StateRef<T> const& src,
                size_type num_valid,
                StreamId stream)
{
    if (src.empty() || num_valid == 0)
    {
        // This attribute is not in use
        dst->clear();
        return;
    }
    dst->resize(num_valid);
    // Copy all items from valid threads
    Copier<T, MemSpace::host> copy{{dst->data(), num_valid}, stream};
    copy(MemSpace::device, {src.data().get(), num_valid});
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Copy to host results from tracks that interacted with a detector.
 */
template<>
void copy_steps<MemSpace::device>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    CELER_EXPECT(output);

    // Store the thread IDs of active tracks that are in a detector
    auto start = thrust::device_pointer_cast(state.valid_id.data().get());
    auto end = thrust::copy_if(
        thrust_execute_on(state.stream_id),
        thrust::make_counting_iterator(size_type(0)),
        thrust::make_counting_iterator(state.size()),
        thrust::device_pointer_cast(state.data.detector.data().get()),
        start,
        HasDetector{});

    // Get the number of threads that are active and in a detector
    size_type num_valid = end - start;

    // Gather the step data on device
    gather_step(state, num_valid);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    copy_field(          \
        &(output->FIELD), state.scratch.FIELD, num_valid, state.stream_id)

    DS_ASSIGN(detector);
    DS_ASSIGN(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_ASSIGN(points[sp].time);
        DS_ASSIGN(points[sp].pos);
        DS_ASSIGN(points[sp].dir);
        DS_ASSIGN(points[sp].energy);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(particle);
    DS_ASSIGN(energy_deposition);
#undef DS_ASSIGN

    // Copies must be complete before returning
    CELER_DEVICE_CALL_PREFIX(
        StreamSynchronize(celeritas::device().stream(state.stream_id).get()));

    CELER_ENSURE(output->detector.size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
