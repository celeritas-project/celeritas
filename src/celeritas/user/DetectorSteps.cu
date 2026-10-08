//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cu
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include <cstddef>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

#if CELERITAS_USE_CUDA
#    include <cub/device/device_select.cuh>
#elif CELERITAS_HAVE_HIPCUB
#    include <hipcub/device/device_select.hpp>
#else
#    include <thrust/copy.h>
#    include <thrust/execution_policy.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "StepData.hh"

#include "detail/StepScratchCopyExecutor.hh"

#if CELERITAS_HAVE_HIPCUB
namespace cub = hipcub;
#endif

using namespace celeritas::literals;

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

using StepStateDeviceRef
    = StepStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
/*!
 * Whether the ID at the given track slot is valid.
 */
template<class IdT>
struct IsValidAt
{
    IdT const* ids;

    CELER_FORCEINLINE_FUNCTION bool operator()(size_type i) const
    {
        return static_cast<bool>(ids[i]);
    }
};

#if CELER_USE_THRUST
//---------------------------------------------------------------------------//
/*!
 * Whether the ID is valid.
 */
struct IsValid
{
    template<class T>
    CELER_FORCEINLINE_FUNCTION bool operator()(OpaqueId<T> const& id) const
    {
        return static_cast<bool>(id);
    }
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Store the slot indices of selected tracks and their count on device.
 *
 * With CUB, the count is written to device memory without synchronizing.
 * The thrust fallback must synchronize to obtain the count.
 */
template<class IdT>
void select_valid_ids(StepStateDeviceRef const& state,
                      StateRef<IdT> const& stencil)
{
    size_type* d_num_selected = state.num_selected.data().get();
    CELER_ASSERT(d_num_selected);
#if CELER_USE_THRUST
    auto start = device_pointer_cast(state.valid_id.data());
    auto end = thrust::copy_if(thrust_execute_on(state.stream_id),
                               thrust::make_counting_iterator(0_sz),
                               thrust::make_counting_iterator(state.size()),
                               device_pointer_cast(stencil.data()),
                               start,
                               IsValid{});
    size_type num_selected = end - start;
    Copier<size_type, MemSpace::device> copy{{d_num_selected, 1},
                                             state.stream_id};
    copy(MemSpace::host, {&num_selected, 1});
    device().stream(state.stream_id).sync();
#else
    auto& stream = device().stream(state.stream_id);
    auto d_in = thrust::make_counting_iterator(0_sz);
    size_type* d_out = state.valid_id.data().get();
    IsValidAt<IdT> is_valid{stencil.data().get()};

    // Calling with nullptr returns the amount of working space needed
    std::size_t temp_storage_bytes = 0;
    // HIP defines hipCUB functions as [[nodiscard]], but we defer error checks
    auto cub_error_code = cub::DeviceSelect::If(nullptr,
                                                temp_storage_bytes,
                                                d_in,
                                                d_out,
                                                d_num_selected,
                                                state.size(),
                                                is_valid,
                                                stream.get());
    CELER_DISCARD(cub_error_code);
    // Allocate temporary storage (stream-ordered)
    DeviceVector<char> temp_storage(temp_storage_bytes, state.stream_id);
    cub_error_code = cub::DeviceSelect::If(temp_storage.data(),
                                           temp_storage_bytes,
                                           d_in,
                                           d_out,
                                           d_num_selected,
                                           state.size(),
                                           is_valid,
                                           stream.get());
    CELER_DISCARD(cub_error_code);
    CELER_DEVICE_API_CALL(PeekAtLastError());
#endif
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
 * Compact selected device step data without synchronizing the stream.
 *
 * Tracks are selected if they interacted with a detector or, if no detectors
 * are used, if their track ID was set during gathering. This enqueues on the
 * state's stream:
 * - the selection of track slots (preserving their order),
 * - the compaction of the selected data into the scratch space, and
 * - an asynchronous copy of the number of selected tracks to \c num_selected.
 *
 * The \c num_selected argument \b must point to pinned host memory that
 * remains valid until the stream reaches this point, and the step data must
 * not be modified before \c copy_compacted_steps is called.
 */
void compact_steps_async(StepStateDeviceRef const& state,
                         size_type* num_selected)
{
    CELER_EXPECT(state);
    CELER_EXPECT(num_selected);

    ScopedProfiling profile_this{"compact-steps"};

    // Store the thread IDs of active tracks that are selected
    if (state.data.detector_id.empty())
    {
        select_valid_ids(state, state.data.track_id);
    }
    else
    {
        select_valid_ids(state, state.data.detector_id);
    }

    // Gather the step data on device, using the count stored on device
    ObserverPtr<size_type const> d_num_selected = state.num_selected.data();
    {
        auto execute_thread
            = detail::StepScratchCopyExecutor{state, d_num_selected};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "gather-step-scratch");
        launch_kernel(state.size(), state.stream_id, execute_thread);
    }

    // Copy the count to the host
    Copier<size_type, MemSpace::host> copy{{num_selected, 1}, state.stream_id};
    copy(MemSpace::device, {d_num_selected.get(), 1});
}

//---------------------------------------------------------------------------//
/*!
 * Copy device step data compacted by \c compact_steps_async to the host.
 *
 * The number of selected tracks must have been copied to the host already,
 * i.e., the stream must have reached the end of \c compact_steps_async . This
 * synchronizes the state's stream.
 */
void copy_compacted_steps(DetectorStepOutput* output,
                          StepStateDeviceRef const& state,
                          size_type num_valid)
{
    CELER_EXPECT(output);
    CELER_EXPECT(num_valid <= state.size());

    ScopedProfiling profile_this{"copy-compacted-steps"};

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    copy_field( \
        &(output->FIELD), state.scratch.FIELD, num_valid, state.stream_id)

    DS_ASSIGN(detector_id);
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
                   state.num_volume_levels,
                   state.stream_id);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(generation);
    DS_ASSIGN(primary_id);
    DS_ASSIGN(post_step_action_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(weight);
    DS_ASSIGN(particle_id);
    DS_ASSIGN(energy_deposition);

    output->num_volume_levels = state.num_volume_levels;

#undef DS_ASSIGN

    // Copies must be complete before returning
    device().stream(state.stream_id).sync();

    CELER_ENSURE(output->size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
}

//---------------------------------------------------------------------------//
/*!
 * Copy to host results from selected tracks.
 *
 * Tracks are selected if they interacted with a detector or, if no detectors
 * are used, if their track ID was set during gathering. This synchronizes the
 * state's stream.
 */
template<>
void copy_steps<MemSpace::device>(DetectorStepOutput* output,
                                  StepStateDeviceRef const& state)
{
    CELER_EXPECT(output);

    ScopedProfiling profile_this{"copy-steps"};

    // Enqueue compaction, then wait for the count to reach the host
    size_type num_valid{0};
    compact_steps_async(state, &num_valid);
    device().stream(state.stream_id).sync();

    copy_compacted_steps(output, state, num_valid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
