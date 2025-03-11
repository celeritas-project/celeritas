//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Filler.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "StepDiagnosticExecutor.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<StepDiagnostic>
StepDiagnostic::make_and_insert(CoreParams const& core,
                                std::string filename_base)
{
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result
        = std::make_shared<StepDiagnostic>(actions.next_id(), aux.next_id());
    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with IDs and filename base.
 *
 * This also writes to the "metadata" JSON suffix.
 */
StepDiagnostic::StepDiagnostic(ActionId action_id, AuxId aux_id)
    : sad_{action_id, "step-diagnostic", "accumulate step statistics"}
    , aux_id_{aux_id}
{
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the statistics and reset them.
 */
StepStatistics StepDiagnostic::GetAndReset(CoreStateInterface& state) const
{
    StepStatistics result;
    auto try_copy = [&](auto* core_state) -> bool {
        if (!core_state)
            return false;

        // Whether the given state is device/host
        constexpr MemSpace M = decltype(*core_state)::memspace;

        // Get the step data from the core state
        auto& step_state = core_state->template aux_data<M>(aux_id_);

        // Stream (i.e., thread) index
        StreamId sid = core_state->stream_id();

        // Copy and reset
        result = this->copy(sid, step_state);
        this->reset(sid, step_state);
        return true;
    };

    if (try_copy(dynamic_cast<CoreState<MemSpace::host>>(&state))) {}
    else if (try_copy(dynamic_cast<CoreState<MemSpace::host>>(&state))) {}
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 *
 * This creates "thread-local" data for the given stream on host or device.
 */
auto StepDiagnostic::create_state(MemSpace m,
                                  StreamId id,
                                  size_type size) const -> UPState
{
    auto result = make_aux_state<StepStateData>(m, id, size);
    CELER_ASSERT(result);
    this->reset(id, *result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Gather data at each step.
 */
void StepDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto& step_state = state.aux_data<MemSpace::native>(aux_id_);

    CoreStateCounters const& counters = state.counters();
    step_state.num_steps += counters.num_active;
    step_state.num_generated += counters.num_generated;
    step_state.num_secondaries += counters.num_secondaries;

    // Create a functor that gathers data from a single track slot
    auto execute
        = make_active_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     StepDiagnosticExecutor{step_state});
    // Run on all track slots
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StepDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#else
extern template class celeritas::Filler<NativeStepStatistics, MemSpace::device>;
#endif

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Copy data from the step state.
 */
template<MemSpace M>
StepStatistics
StepDiagnostic::copy(StreamId sid,
                     StepStateData<M, Ownership::reference>& step_state) const
{
    // Copy from device (or host) to host
    NativeStepStatistics copied;
    Copier<int, MemSpace::host> copy_to_host{Span{&copied, 1}, sid};
    copy_to_host(M, step_state.data);

    // Save to output
    StepStatistics result;
    result.step_length = copied.step_length;
    result.energy_deposited = copied.energy_deposited;
    result.num_steps = step_state.host_data.steps;
    result.num_primaries = step_state.host_data.generated
                           - step_state.host_data.secondaries;
    result.num_secondaries = step_state.host_data.secondaries;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the accumulated state.
 */
template<MemSpace M>
void StepDiagnostic::reset(
    StreamId sid, StepStateData<M, Ownership::reference>& step_state) const
{
    Filler<NativeStepStatistics, M> fill_empty({0.0, 0.0}, sid);
    fill_empty(step_state.data);
    step_state.host_data = {};
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
