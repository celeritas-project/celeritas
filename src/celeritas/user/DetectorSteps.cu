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
/*!
 * Resize the destination and enqueue a copy of the first items of a field.
 *
 * The destination is cleared if the field is unused or empty.
 */
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                T const* src,
                size_type count,
                Stream const& stream)
{
    if (!src || count == 0)
    {
        // This field is not in use or had no hits
        dst->clear();
        return;
    }
    dst->resize(count);
    // Copy all items from valid threads
    copy_bytes(MemSpace::host,
               dst->data(),
               MemSpace::device,
               src,
               count * sizeof(T),
               stream);
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                StateRef<T> const& src,
                size_type num_valid,
                Stream const& stream)
{
    copy_field(
        dst, src.empty() ? nullptr : src.data().get(), num_valid, stream);
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                ItemRef<T> const& src,
                size_type num_valid,
                size_type per_thread,
                Stream const& stream)
{
    CELER_EXPECT(per_thread > 0 || src.empty());
    copy_field(dst,
               src.empty() ? nullptr : src.data().get(),
               num_valid * per_thread,
               stream);
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
 * The pinned host memory referenced by \c num_selected must remain valid
 * until the stream reaches this point, and the step data must not be modified
 * before \c copy_compacted_steps is called.
 */
void compact_steps_async(StepStateDeviceRef const& state,
                         AsyncResultRef<size_type> num_selected)
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
    auto* h_num_selected = static_cast<size_type*>(num_selected.ptr);
    Copier<size_type, MemSpace::host> copy{{h_num_selected, 1},
                                           state.stream_id};
    copy(MemSpace::device, {d_num_selected.get(), 1});
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a copy of compacted device step data to the host.
 *
 * The number of selected tracks must have been copied to the host already,
 * i.e., the state's stream must have reached the end of
 * \c compact_steps_async . The output is resized immediately, and copies from
 * the state's scratch space are enqueued on the given stream without
 * synchronizing. If \c stream is not the state's stream, it must wait on an
 * event recorded after the compaction.
 *
 * The output must not be read or resized, and the scratch space must not be
 * overwritten, until the stream reaches this point.
 */
void copy_compacted_steps_async(DetectorStepOutput* output,
                                StepStateDeviceRef const& state,
                                size_type num_valid,
                                Stream const& stream)
{
    CELER_EXPECT(output);
    CELER_EXPECT(num_valid <= state.size());

    ScopedProfiling profile_this{"copy-compacted-steps"};

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    copy_field(&(output->FIELD), state.scratch.FIELD, num_valid, stream)

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
                   stream);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(primary_id);
    DS_ASSIGN(post_step_action_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(weight);
    DS_ASSIGN(particle_id);
    DS_ASSIGN(energy_deposition);

    output->num_volume_levels = state.num_volume_levels;

#undef DS_ASSIGN

    CELER_ENSURE(output->size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
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
    auto& stream = device().stream(state.stream_id);
    copy_compacted_steps_async(output, state, num_valid, stream);

    // Copies must be complete before returning
    stream.sync();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
