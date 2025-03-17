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
namespace
{
//---------------------------------------------------------------------------//
template<class A, class B>
using DerivedPtr =
    typename std::conditional<std::is_const<A>::value, B const*, B*>::type;

// TODO: this helper function should be moved to corecel
template<template<MemSpace> class Derived, typename Base, typename Func>
auto visit_memspace_derived(Base& base, Func&& apply_derived)
{
    if (auto* derived
        = dynamic_cast<DerivedPtr<Base, Derived<MemSpace::host>>>(&base))
    {
        return apply_derived(*derived);
    }
    else if (auto* derived
             = dynamic_cast<DerivedPtr<Base, Derived<MemSpace::device>>>(&base))
    {
        return apply_derived(*derived);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<StepDiagnostic>
StepDiagnostic::make_and_insert(CoreParams const& core)
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
 * Construct with IDs.
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

    // Since the underlying state (and thus the copy methods we have to call)
    // is host/device-dependent, we use a helper lambda
    visit_memspace_derived<CoreState>(state, [&](auto& derived) {
        // Whether the given state is device/host
        constexpr MemSpace M
            = std::remove_reference_t<decltype(derived)>::memspace;
        CELER_LOG(debug) << "Copying step diagnostics from " << to_cstring(M);

        // Get the step data from the core state
        auto& step_state = derived.template aux_data<StepStateData>(aux_id_);
        // Copy it
        copy_to_host(step_state.data, Span{&data, 1}, derived.stream_id());
        host_data = step_state.host_data;
        // Zero for the next event
        reset(&step_state, derived.stream_id());
        return true;
    });

    // Save to output, converting units
    StepStatistics result;
    result.step_length = convert_to_geant(data.step_length, clhep_length);
    result.energy_deposition = data.energy_deposition;
    result.num_steps = host_data.steps;
    result.num_primaries = host_data.generated;
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

    // Accumulate counters
    this->accum_counters(state.counters(), step_state.host_data);

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
/*!
 * Accumulate step/track counters.
 */
void StepDiagnostic::accum_counters(CoreStateCounters const& counters,
                                    HostStepStatistics& stats) const
{
    stats.steps += counters.num_active;
    stats.generated += counters.num_generated;
    stats.secondaries += counters.num_secondaries;
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
