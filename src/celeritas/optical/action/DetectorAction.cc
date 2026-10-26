//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/AuxParamsRegistry.hh"  // IWYU pragma: keep
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/Collection.hh"
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
 * The device scratch and pinned hit buffers are allocated and sized here once
 * per stream so that \c step never needs to reallocate.
 */
auto DetectorAction::create_state(
    MemSpace m, StreamId stream, size_type size) const -> UPState
{
    CELER_EXPECT(size > 0);

    auto result = std::make_unique<DetectorActionState>();
    if (m == MemSpace::device)
    {
        result->hits.emplace(size, stream);
    }

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

    std::size_t num_valid = [&all_hits] {
        ScopedProfiling profile_this("prune");
        auto end
            = std::remove_if(all_hits.begin(), all_hits.end(), LogicalNot{});
        return std::distance(all_hits.begin(), end);
    }();

    this->callback_hits(make_span(all_hits).first(num_valid));
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
 * Process the hits remaining from device in step order.
 */
void DetectorAction::end_transport(CoreParams const&,
                                   CoreStateDevice& state) const
{
    auto& aux_state = get<DetectorActionState>(*state.aux(), aux_id_);
    if (aux_state.hits)
    {
        this->flush_hits(*aux_state.hits);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process the hits of the previous steps and copy the hits of the last.
 *
 * The hits copied during the previous step (i.e., from two steps ago) are
 * sent to the callback, and the copy of the hits compacted in the previous
 * step is enqueued. Because the transport loop waits for each step before
 * starting the next, the compaction has completed and this does not wait.
 * The returned buffer is ready for compacting the hits of the current step.
 */
auto DetectorAction::copy_hits_async(CoreStateDevice& state) const -> Hits&
{
    if constexpr (!CELER_USE_DEVICE)
    {
        // Mark unreachable for optimization and coverage
        CELER_ASSERT_UNREACHABLE();
    }

    auto& aux_state = get<DetectorActionState>(*state.aux(), aux_id_);
    CELER_ASSERT(aux_state.hits);

    auto const num_slots = state.ref().detectors.detector_hits.size();
    if (CELER_UNLIKELY(aux_state.hits->capacity() != num_slots))
    {
        // FIXME: we currently share a single aux state between core and optical
        // tracking loops, but they may have different number of tracks, so this
        // is an error. It basically amounts to doing a lazy resize.
        CELER_LOG(warning) << "FIXME: expected optical aux state to be size "
                           << num_slots << " but is actually size "
                           << aux_state.hits->capacity();
        this->flush_hits(*aux_state.hits);
        aux_state.hits.emplace(num_slots, state.stream_id());
    }

    auto& hits = *aux_state.hits;
    if (hits.has_compacted())
    {
        // The pinned output is reused by the next copy, so process it first
        this->callback_hits(hits.take_copied());
        hits.copy_async();
    }
    CELER_ENSURE(!hits.has_compacted());
    return hits;
}

//---------------------------------------------------------------------------//
/*!
 * Process all hits remaining from device in step order.
 */
void DetectorAction::flush_hits(Hits& hits) const
{
    this->callback_hits(hits.take_copied());
    if (hits.has_compacted())
    {
        hits.copy_async();
        this->callback_hits(hits.take_copied());
    }
    CELER_ENSURE(!hits.has_compacted() && !hits.has_copied());
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
