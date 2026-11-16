//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/StepProcessor.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include "StepProcessor.hh"

#include <exception>
#include <utility>
#include <G4Step.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TraceCounter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct the shared step and track reconstruction.
 *
 * If a device is active, the event for copying device steps is also created.
 */
template<class D, class O, class H, class S>
StepProcessor<D, O, H, S>::StepProcessor(VecParticle const& particles)
{
    step_ = GeantTrackReconstruction::make_g4step();
    CELER_ASSERT(step_);
    if (!particles.empty())
    {
        // Reconstruct track
        track_reconstruction_
            = std::make_shared<GeantTrackReconstruction>(particles, step_);
    }

    if (celeritas::device())
    {
        // Create the event for copying steps
        copied_ = DeviceEvent{celeritas::device()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Wait for any pending device compaction or copy before releasing buffers.
 *
 * The count of selected steps and the compacted steps are asynchronously
 * copied into pinned memory owned by this class, so pending copies must
 * complete before that memory is freed (e.g., when an exception unwinds the
 * transporter mid-step).
 */
template<class D, class O, class H, class S>
StepProcessor<D, O, H, S>::~StepProcessor()
{
    try
    {
        if (this->has_pending_steps())
        {
            num_selected_.sync();
            copied_.sync();
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG_LOCAL(error) << "Failed to wait for pending step compaction "
                                  "or copy while destroying processor: "
                               << e.what();
    }
    catch (...)
    {
        CELER_LOG_LOCAL(error) << "Failed to wait for pending step compaction "
                                  "or copy while destroying processor: "
                                  "unknown exception";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Allocate pinned host output for the largest possible device step.
 *
 * This \b must be called during setup, once the device step state exists and
 * before any device steps are processed, so that no pinned memory is
 * allocated while stepping: growing the output later would call
 * \c cudaHostAlloc and \c cudaFreeHost , which serialize CUDA API calls from
 * all host threads.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::reserve(StepStateDeviceRef const& states)
{
    CELER_EXPECT(states);
    CELER_EXPECT(!this->has_pending_steps());

    reserve_compacted_steps(&steps_, states);
    steps_reserved_ = true;
}

//---------------------------------------------------------------------------//
/*!
 * Process steps generated on the CPU.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::operator()(StepStateHostRef const& states)
{
    copy_steps(&steps_, states);
    this->process_local_steps();
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue compaction of device step data without synchronizing.
 *
 * The selection and compaction into the state's scratch space are enqueued on
 * the state's stream, along with a copy of the selected count to pinned host
 * memory. Deferring the data copy keeps host-side reconstruction and its
 * stream synchronization out of the device action sequence.
 *
 * The compaction waits on the device for any previous copy of the scratch
 * space, so the step that produces this data can be launched before that copy
 * completes. The pinned host output must have been allocated with \c reserve .
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::operator()(StepStateDeviceRef const& states)
{
    CELER_EXPECT(states);
    CELER_EXPECT(this->scratch_state() == ScratchState::free);
    CELER_EXPECT(copied_);
    CELER_EXPECT(steps_reserved_);

    auto& stream = celeritas::device().stream(states.stream_id);
    // Don't overwrite the scratch space while it is being copied
    stream.wait(copied_);
    compact_steps_async(states, num_selected_);
    pending_device_steps_ = states;
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue the copy of compacted device steps after their step completes.
 *
 * This waits for the compaction enqueued by the call operator, which has
 * already completed if the caller established device step completion. Any
 * previously copied steps are processed first since they share host storage.
 * The copy is enqueued on the memcpy stream without synchronizing.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::copy_pending_steps()
{
    if (this->scratch_state() == ScratchState::free)
    {
        return;
    }

    // The copy overwrites the output, so finish processing it first
    this->process_copied_steps();
    CELER_ASSERT(output_ == OutputState::empty);

    // The count is loaded after the compaction, so the host-observed load
    // also orders the copy after the compaction. Keep the steps pending until
    // then so the destructor waits on failure.
    num_selected_.sync();
    auto states = std::exchange(pending_device_steps_, {});
    auto& stream = celeritas::device().memcpy_stream(states.stream_id);
    try
    {
        copy_compacted_steps_async(
            &steps_, states, num_selected_.value(), stream);
    }
    catch (...)
    {
        // Don't leave partially enqueued copies writing into host buffers
        CELER_LOG_LOCAL(critical)
            << "Failed to enqueue copy of compacted steps: waiting for "
               "already enqueued copies before propagating the error";
        stream.sync();
        throw;
    }
    copied_.record(stream);
    output_ = OutputState::copied;
}

//---------------------------------------------------------------------------//
/*!
 * Wait for and process copied device steps.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::process_copied_steps()
{
    if (output_ == OutputState::empty)
    {
        return;
    }

    copied_.sync();
    output_ = OutputState::empty;
    this->process_local_steps();
}

//---------------------------------------------------------------------------//
/*!
 * Process copied device steps only if the copy has completed.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::process_copied_steps_if_ready()
{
    if (output_ == OutputState::copied && copied_.ready())
    {
        this->process_copied_steps();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call steps from an output.
 *
 * In an application setting, this is always called with our local data \c
 * steps_ as an argument. For tests, we can call this function explicitly using
 * local test data. Each step is dispatched to the derived class.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::operator()(StepOutput const& out) const
{
    ScopedProfiling profile_this{D::process_label};
    trace_counter(D::process_label, out.size());
    D const& derived = this->derived();
    for (auto i : range(out.size()))
    {
        derived(out, i);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process steps already copied into temporary CPU storage.
 */
template<class D, class O, class H, class S>
void StepProcessor<D, O, H, S>::process_local_steps()
{
    if (steps_)
    {
        num_hits_ += steps_.size();
        static_cast<StepProcessor const&>(*this)(steps_);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
