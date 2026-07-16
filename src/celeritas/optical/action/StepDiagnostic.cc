//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/StepDiagnosticExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<StepDiagnostic>
StepDiagnostic::make_and_insert(CoreParams const& core, size_type max_step_bin)
{
    ActionRegistry& actions = *core.action_reg();
    OutputRegistry& out = *core.output_reg();
    auto result = std::make_shared<StepDiagnostic>(
        actions.next_id(), max_step_bin, core.sizes().streams);
    actions.insert(result);
    out.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with ID and counts.
 */
StepDiagnostic::StepDiagnostic(ActionId id,
                               size_type max_step_bin,
                               size_type num_streams)
    : StepDiagnosticBase(1, max_step_bin, num_streams), id_(id)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string_view StepDiagnostic::description() const
{
    return "accumulate total optical step counters";
}

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void StepDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::StepDiagnosticExecutor{
                                  this->store().params<MemSpace::native>(),
                                  this->store().state<MemSpace::native>(
                                      state.stream_id(), this->state_size())}};
    return launch_action(state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StepDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
