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

#include "corecel/data/Copier.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "StepData.hh"

#include "detail/StepScratchCopyExecutor.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::native>;

template<class T>
using ItemRef
    = celeritas::Collection<T, Ownership::reference, MemSpace::native>;

//---------------------------------------------------------------------------//
struct HasDetector
{
    CELER_FORCEINLINE_FUNCTION bool operator()(DetectorId const& d)
    {
        return static_cast<bool>(d);
    }
};

//---------------------------------------------------------------------------//
size_type count_num_valid(
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    // Store the thread IDs of active tracks that are in a detector
    auto start = device_pointer_cast(state.valid_id.data());
    auto end = thrust::copy_if(thrust_execute_on(state.stream_id),
                               thrust::make_counting_iterator(size_type(0)),
                               thrust::make_counting_iterator(state.size()),
                               device_pointer_cast(state.data.detector.data()),
                               start,
                               HasDetector{});
    return end - start;
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                StateRef<T> const& src,
                size_type num_valid,
                StreamId stream)
{
    if (src.empty() || num_valid == 0)
    {
        // This field is not in use or had no hits
        dst->clear();
        return;
    }
    dst->resize(num_valid);
    // Copy all items from valid threads
    Copier<T, MemSpace::host> copy{{dst->data(), num_valid}, stream};
    copy(MemSpace::device, {src.data().get(), num_valid});
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                ItemRef<T> const& src,
                size_type num_valid,
                size_type per_thread,
                StreamId stream)
{
    CELER_EXPECT(per_thread > 0 || src.empty());
    if (src.empty() || num_valid == 0)
    {
        // This attribute is not in use
        dst->clear();
        return;
    }
    dst->resize(num_valid * per_thread);
    // Copy all items from valid threads
    Copier<T, MemSpace::host> copy{{dst->data(), num_valid * per_thread},
                                   stream};
    copy(MemSpace::device, {src.data().get(), num_valid * per_thread});
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

    ScopedProfiling profile_this{"copy-steps"};

    // Get the number of threads that are active and in a detector
    size_type const num_valid = count_num_valid(state);

    // Gather the step data on device
    {
        auto execute_thread = detail::StepScratchCopyExecutor{state, num_valid};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "gather-step-scratch");
        launch_kernel(num_valid, state.stream_id, execute_thread);
    }

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

        copy_field(&(output->points[sp].volume_instance_ids),
                   state.scratch.points[sp].volume_instance_ids,
                   num_valid,
                   state.volume_instance_depth,
                   state.stream_id);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(particle);
    DS_ASSIGN(energy_deposition);

    output->volume_instance_depth = state.volume_instance_depth;

#undef DS_ASSIGN

    // Copies must be complete before returning
    CELER_DEVICE_API_CALL(
        StreamSynchronize(celeritas::device().stream(state.stream_id).get()));

    CELER_ENSURE(output->detector.size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
