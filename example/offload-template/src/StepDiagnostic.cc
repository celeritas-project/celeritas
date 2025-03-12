//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include <type_traits>
#include <vector>
#include <celeritas/global/ActionInterface.hh>
#include <celeritas/global/ActionLauncher.hh>
#include <celeritas/global/CoreParams.hh>
#include <celeritas/global/CoreState.hh>
#include <celeritas/global/TrackExecutor.hh>
#include <celeritas/track/CoreStateCounters.hh>
#include <corecel/Macros.hh>
#include <corecel/data/AuxParamsRegistry.hh>
#include <corecel/data/AuxStateVec.hh>
#include <corecel/data/Collection.hh>
#include <corecel/data/CollectionAlgorithms.hh>
#include <corecel/data/PinnedAllocator.hh>
#include <corecel/io/Logger.hh>
#include <corecel/sys/ActionRegistry.hh>
#include <geocel/g4/Convert.hh>

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

    // Set up shared data on host and device
    mirror_ = CollectionMirror{HostVal<StepParamsData>{}};
}

//---------------------------------------------------------------------------//
/*!
 * Get the statistics and reset them.
 */
StepStatistics StepDiagnostic::GetAndReset(CoreStateInterface& state) const
{
    // Kernel-collected statistics copied to host memory
    NativeStepStatistics data;
    HostStepStatistics host_data;

    auto try_copy = [&](auto* core_state) -> bool {
        if (!core_state)
            return false;

        // Whether the given state is device/host
        constexpr MemSpace M
            = std::remove_reference_t<decltype(*core_state)>::memspace;
        CELER_LOG(debug) << "Copying step diagnostics from " << to_cstring(M);

        // Get the step data from the core state
        auto& step_state
            = core_state->template aux_data<StepStateData>(aux_id_);
        // Copy it
        copy_to_host(step_state.data, Span{&data, 1}, core_state->stream_id());
        host_data = step_state.host_data;
        // Zero for the next event
        reset(&step_state, core_state->stream_id());
        return true;
    };

    if (try_copy(dynamic_cast<CoreState<MemSpace::host>*>(&state))) {}
    else if (try_copy(dynamic_cast<CoreState<MemSpace::device>*>(&state))) {}
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }

    // Save to output, converting units
    StepStatistics result;
    result.step_length = convert_to_geant(data.step_length, clhep_length);
    result.energy_deposition = data.energy_deposition;
    result.num_steps = host_data.steps;
    result.num_primaries = host_data.generated - host_data.secondaries;
    result.num_secondaries = host_data.secondaries;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 *
 * This creates and initializes "thread-local" data for the given stream on
 * host or device.
 */
auto StepDiagnostic::create_state(MemSpace m,
                                  StreamId id,
                                  size_type size) const -> UPState
{
    auto result = make_aux_state<StepStateData>(*this, m, id, size);
    CELER_ASSERT(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Gather data at each step.
 */
void StepDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto const& step_params = this->ref<MemSpace::native>();
    auto& step_state = state.aux_data<StepStateData>(aux_id_);

    CoreStateCounters const& counters = state.counters();
    step_state.host_data.steps += counters.num_active;
    step_state.host_data.generated += counters.num_generated;
    step_state.host_data.secondaries += counters.num_secondaries;

    // Create a functor that gathers data from a single track slot
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        StepDiagnosticExecutor{step_params, step_state});
    // Run on all track slots
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StepDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
