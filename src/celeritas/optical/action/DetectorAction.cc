//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"  // IWYU pragma: keep
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/optical/CoreParams.hh"  // IWYU pragma: keep
#include "celeritas/optical/CoreState.hh"  // IWYU pragma: keep

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/DetectorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Create and add to optical params.
 */
std::shared_ptr<DetectorAction> DetectorAction::make_and_insert(
    SPActionRegistry const& action_reg,
    SPAuxParamsRegistry const& aux_reg,
    CallbackFunc const& cb)
{
    CELER_EXPECT(action_reg);
    CELER_EXPECT(aux_reg);
    CELER_EXPECT(cb);
    auto action = std::make_shared<DetectorAction>(
        action_reg->next_id(), aux_reg->next_id(), cb);
    action_reg->insert(action);
    aux_reg->insert(action);
    return action;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, aux ID, and callback function.
 */
DetectorAction::DetectorAction(
    ActionId aid, AuxId aux_id, CallbackFunc const& callback)
    : sad_(aid, "detector", "Score optical detector hits")
    , aux_id_(aux_id)
    , callback_(callback)
{
    CELER_EXPECT(callback);
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Build auxiliary state data for a stream.
 *
 * The pinned hit buffer is allocated and sized here once per stream so that
 * \c step and \c load_hits_sync never need to reallocate.
 */
auto DetectorAction::create_state(MemSpace, StreamId, size_type size) const
    -> UPState
{
    CELER_EXPECT(size > 0);

    auto result = std::make_unique<DetectorActionState>();
    result->hits.resize(size);

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Launch the detector action on host.
 */
void DetectorAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::DetectorExecutor{state.ref().detectors}};
    launch_action(state, execute);

    auto all_hits
        = state.ref().detectors.detector_hits[AllItems<DetectorHit,
                                                       MemSpace::host>{}];

    auto& temp_hits = get<DetectorActionState>(*state.aux(), aux_id_).hits;
    CELER_ASSERT(temp_hits.size() == all_hits.size());

    std::size_t num_valid = [&all_hits, &temp_hits] {
        ScopedProfiling profile_this("copy-prune");
        // Copy all valid hits into the persistent buffer, keeping only the
        // valid ones
        auto end = std::copy_if(
            all_hits.begin(), all_hits.end(), temp_hits.begin(), Identity{});
        return std::distance(temp_hits.begin(), end);
    }();

    this->callback_hits(make_span(temp_hits).first(num_valid));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void DetectorAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 */
Span<DetectorHit const> DetectorAction::load_hits_sync(
    CoreStateDevice const& state) const
{
    auto const& device_hits = state.ref().detectors.detector_hits;
    auto& temp_hits = get<DetectorActionState>(*state.aux(), aux_id_).hits;
    CELER_ASSERT(temp_hits.size() == device_hits.size());

    {
        // Copy all track hits to host from device
        ScopedProfiling profile_this("copy");
        copy_to_host(device_hits, make_span(temp_hits), state.stream_id());

        // Ensure copy is complete
        celeritas::device().stream(state.stream_id()).sync();
    }

    // Move all hits with a valid detector ID to the front of the buffer,
    // keeping the buffer itself at its original size
    std::size_t num_valid = [&temp_hits] {
        ScopedProfiling profile_this("prune");
        auto end
            = std::remove_if(temp_hits.begin(), temp_hits.end(), LogicalNot{});
        return std::distance(temp_hits.begin(), end);
    }();
    CELER_ASSERT(num_valid <= temp_hits.size());

    return make_span(temp_hits).first(num_valid);
}

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * Copied hits might be invalid, and are removed before sending into the
 * callback function. The callback is only executed when a non-zero number of
 * valid hits occurs.
 */
void DetectorAction::callback_hits(Span<DetectorHit const> hits) const
{
    // Send hits to the callback function, if there are any
    if (!hits.empty())
    {
        ScopedProfiling profile_this("callback");
        callback_(hits);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
