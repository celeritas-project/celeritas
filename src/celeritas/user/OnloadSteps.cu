//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/OnloadSteps.cu
//---------------------------------------------------------------------------//
#include "OnloadSteps.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"

#include "OnloadData.hh"

#include "detail/OnloadScratchCopyExecutor.hh"
#include "detail/StepCopyUtils.device.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
using OnloadStateDeviceRef
    = OnloadStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Compact selected device step data without synchronizing the stream.
 *
 * Tracks are selected if their track ID was set during gathering. This
 * enqueues on the state's stream:
 * - the selection of track slots (preserving their order),
 * - the compaction of the selected data into the scratch space, and
 * - an asynchronous load of the number of selected tracks into
 *   \c num_selected .
 *
 * The snapshot must not be destroyed until its load completes, which also
 * implies that the compaction has completed. The scratch space must not be
 * overwritten until the copy enqueued by \c copy_compacted_steps_async
 * completes.
 */
void compact_steps_async(OnloadStateDeviceRef const& state,
                         AsyncSnapshot<size_type>& num_selected)
{
    CELER_EXPECT(state);

    ScopedProfiling profile_this{"compact-onload-steps"};

    // Store the thread IDs of active tracks that are selected
    detail::select_valid_ids(state, state.data.track_id);

    // Gather the step data on device, using the count stored on device
    ObserverPtr<size_type const> d_num_selected = state.num_selected.data();
    {
        auto execute_thread
            = detail::OnloadScratchCopyExecutor{state, d_num_selected};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "onload-step-scratch");
        launch_kernel(state.size(), state.stream_id, execute_thread);
    }

    // Copy the count to the host
    num_selected.load_async(d_num_selected, state.stream_id);
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a copy of compacted device step data to the host.
 *
 * The number of selected tracks must have been loaded to the host already,
 * i.e., the snapshot passed to \c compact_steps_async must be ready. Since
 * that load follows the compaction, the copies may be enqueued on any stream.
 * The output is resized immediately, and copies from the state's scratch
 * space are enqueued on the given stream without synchronizing.
 *
 * The output must not be read or resized, and the scratch space must not be
 * overwritten, until the stream reaches this point.
 */
void copy_compacted_steps_async(OnloadStepOutput* output,
                                OnloadStateDeviceRef const& state,
                                size_type num_valid,
                                Stream const& stream)
{
    CELER_EXPECT(output);
    CELER_EXPECT(num_valid <= state.size());

    ScopedProfiling profile_this{"copy-compacted-steps"};

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    detail::copy_field( \
        &(output->FIELD), state.scratch.FIELD, num_valid, stream)

    DS_ASSIGN(track_id);
    DS_ASSIGN(energy);
    DS_ASSIGN(time);
    DS_ASSIGN(pos);
    DS_ASSIGN(dir);

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
void reserve_compacted_steps(OnloadStepOutput* output,
                             OnloadStateDeviceRef const& state)
{
    CELER_EXPECT(output);
    CELER_EXPECT(state);

#define DS_RESERVE(FIELD) \
    detail::reserve_field(&(output->FIELD), state.scratch.FIELD)

    DS_RESERVE(track_id);
    DS_RESERVE(energy);
    DS_RESERVE(time);
    DS_RESERVE(pos);
    DS_RESERVE(dir);

#undef DS_RESERVE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
