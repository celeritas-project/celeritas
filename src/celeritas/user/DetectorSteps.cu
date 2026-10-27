//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cu
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include <thrust/iterator/counting_iterator.h>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"

#include "StepData.hh"

#include "detail/StepScratchCopyExecutor.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::native>;

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
    copy_if_async(thrust::make_counting_iterator(0_sz),
                  state.valid_id.data(),
                  state.size(),
                  state.num_selected.data(),
                  IsValidAt<IdT>{stencil.data().get()},
                  state.stream_id);
}

//---------------------------------------------------------------------------//
/*!
 * Resize the destination and enqueue a copy of the first items of a field.
 *
 * The field may be indexed by track slot or, for multi-level data, by item.
 * The destination is cleared if the field is unused or had no hits.
 */
template<class T, class I>
void copy_field(
    DetectorStepOutput::PinnedVec<T>* dst,
    Collection<T, Ownership::reference, MemSpace::native, I> const& src,
    size_type count,
    Stream const& stream)
{
    if (src.empty() || count == 0)
    {
        // This field is not in use or had no hits
        dst->clear();
        return;
    }
    CELER_EXPECT(count <= src.size());
    dst->resize(count);
    // Copy all items from valid threads. Note that make_observer would give
    // a device pointer here since "native" memory is the device in .cu files.
    copy_bytes<MemSpace::host, MemSpace::device>(
        ObserverPtr<T, MemSpace::host>{dst->data()},
        src.data(),
        count * sizeof(T),
        stream);
}

//---------------------------------------------------------------------------//
/*!
 * Reserve the destination for every item of a field if the field is in use.
 */
template<class T, class SrcRef>
void reserve_field(DetectorStepOutput::PinnedVec<T>* dst, SrcRef const& src)
{
    if (!src.empty())
    {
        dst->reserve(src.size());
    }
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
 * - an asynchronous load of the number of selected tracks into
 *   \c num_selected .
 *
 * The snapshot must not be read or destroyed until the stream reaches this
 * point, and the scratch space must not be overwritten until the copy
 * enqueued by \c copy_compacted_steps_async completes.
 *
 * \warning The scratch space is shared by all \c StepInterface callbacks of a
 * \c StepCollector, but nothing tracks its outstanding copies: each caller
 * can only order its compaction after its \em own copies. Only one callback
 * per collector may therefore compact step data with this function if any
 * caller defers its copy (currently only \c GeantSd , through
 * \c detail::HitProcessor ).
 */
void compact_steps_async(StepStateDeviceRef const& state,
                         AsyncSnapshot<size_type>& num_selected)
{
    CELER_EXPECT(state);

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
    num_selected.load_async(d_num_selected, state.stream_id);
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

        // Volume instances are stored per track and level
        auto const& vids = state.scratch.points[sp].volume_instance_ids;
        CELER_ASSERT(state.num_volume_levels > 0 || vids.empty());
        copy_field(&(output->points[sp].volume_instance_ids),
                   vids,
                   num_valid * state.num_volume_levels,
                   stream);
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

    CELER_ENSURE(output->size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
}

//---------------------------------------------------------------------------//
/*!
 * Reserve output capacity for the largest possible compacted step.
 *
 * Each field in use is reserved to the size of its scratch space, which bounds
 * the number of items \c copy_compacted_steps_async can copy, so later copies
 * into the output never reallocate it. Reallocating pinned memory calls
 * \c cudaHostAlloc and \c cudaFreeHost, which serialize CUDA API calls from
 * all host threads.
 *
 * The output must not be the destination of a copy in flight.
 */
void reserve_compacted_steps(DetectorStepOutput* output,
                             StepStateDeviceRef const& state)
{
    CELER_EXPECT(output);
    CELER_EXPECT(state);

#define DS_RESERVE(FIELD) reserve_field(&(output->FIELD), state.scratch.FIELD)

    DS_RESERVE(detector_id);
    DS_RESERVE(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_RESERVE(points[sp].time);
        DS_RESERVE(points[sp].pos);
        DS_RESERVE(points[sp].dir);
        DS_RESERVE(points[sp].energy);
        DS_RESERVE(points[sp].volume_instance_ids);
    }

    DS_RESERVE(event_id);
    DS_RESERVE(parent_id);
    DS_RESERVE(primary_id);
    DS_RESERVE(post_step_action_id);
    DS_RESERVE(track_step_count);
    DS_RESERVE(step_length);
    DS_RESERVE(weight);
    DS_RESERVE(particle_id);
    DS_RESERVE(energy_deposition);

#undef DS_RESERVE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
