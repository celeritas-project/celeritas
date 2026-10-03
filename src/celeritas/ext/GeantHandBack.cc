//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantHandBack.cc
//---------------------------------------------------------------------------//
#include "GeantHandBack.hh"

#include <utility>

#include "corecel/Assert.hh"

#include "detail/HandBackProcessor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and number of streams.
 */
GeantHandBack::GeantHandBack(Input const& inp, StreamId::size_type num_streams)
    : input_{inp}, processors_{num_streams}
{
    CELER_EXPECT(num_streams > 0);
}

//---------------------------------------------------------------------------//
//! Default destructor
GeantHandBack::~GeantHandBack() = default;

//---------------------------------------------------------------------------//
/*!
 * Create local hand-back processor.
 *
 * Due to Geant4 multithread semantics, this \b must be done on the same CPU
 * thread on which the resulting processor is used, since the processor
 * allocates Geant4 tracks.
 */
auto GeantHandBack::make_local_processor(
    StreamId sid, SPTrackReconstruction recon) -> SPProcessor
{
    CELER_EXPECT(sid < processors_.size());
    CELER_EXPECT(recon);

    return processors_.make(sid, [this, &recon] {
        return new HandBackProcessor(std::move(recon), input_.locate_touchable);
    });
}

//---------------------------------------------------------------------------//
/*!
 * Select only handed-back tracks.
 */
auto GeantHandBack::filters() const -> Filters
{
    Filters result;
    result.hand_back = true;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Selection of data required to reconstruct tracks.
 */
StepSelection GeantHandBack::selection() const
{
    StepSelection result;
    auto& post = result.points[StepPoint::post];
    post.time = true;
    post.pos = true;
    post.dir = true;
    post.energy = true;
    post.volume_instance_ids = input_.locate_touchable;

    result.parent_id = true;
    result.primary_id = true;
    result.post_step_action_id = true;
    result.weight = true;
    result.particle_id = true;
    result.hand_back_reason = true;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reconstruct CPU-generated handed-back tracks.
 */
void GeantHandBack::process_steps(HostStepState state)
{
    processors_.get(state.stream_id)(state.steps);
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue compaction of device-generated handed-back tracks.
 *
 * The thread-local processor completes the transfer and reconstruction after
 * the asynchronous step result is ready.
 */
void GeantHandBack::process_steps(DeviceStepState state)
{
    processors_.get(state.stream_id)(state.steps);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
