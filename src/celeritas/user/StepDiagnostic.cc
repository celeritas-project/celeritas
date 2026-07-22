//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

#include "detail/StepDiagnosticExecutor.hh"

namespace celeritas
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
        actions.next_id(), core.particle(), max_step_bin, core.sizes().streams);
    actions.insert(result);
    out.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with particle data.
 */
StepDiagnostic::StepDiagnostic(ActionId id,
                               SPConstParticle particle,
                               size_type max_step_bin,
                               size_type num_streams)
    : StepDiagnosticBase(particle->size(), max_step_bin, num_streams), id_(id)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(particle);
}

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string_view StepDiagnostic::description() const
{
    return "accumulate total step counters";
}

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void StepDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepDiagnosticExecutor{
            this->store().params<MemSpace::native>(),
            this->store().state<MemSpace::native>(state.stream_id(),
                                                  this->state_size())});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StepDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
