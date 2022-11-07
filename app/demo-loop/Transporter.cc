//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <csignal>
#include <memory>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/VectorUtils.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/global/ActionRegistry.hh" // IWYU pragma: keep
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/global/detail/ActionSequence.hh"

#include "diagnostic/EnergyDiagnostic.hh"
#include "diagnostic/ParticleProcessDiagnostic.hh"
#include "diagnostic/StepDiagnostic.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES AND FUNCTIONS
//---------------------------------------------------------------------------//
template<MemSpace M>
using MemTag = std::integral_constant<MemSpace, M>;

DiagnosticStore::VecUPDiag<MemSpace::host>&
get_diag_ref(DiagnosticStore& params, MemTag<MemSpace::host>)
{
    return params.host;
}

DiagnosticStore::VecUPDiag<MemSpace::device>&
get_diag_ref(DiagnosticStore& params, MemTag<MemSpace::device>)
{
    return params.device;
}

//---------------------------------------------------------------------------//
//! Adapt a vector of diagnostics to the Action interface
class DiagnosticActionAdapter final : public ExplicitActionInterface
{
  public:
    using SPDiagnostics = std::shared_ptr<DiagnosticStore>;

  public:
    DiagnosticActionAdapter(ActionId id, SPDiagnostics diag)
        : id_(id), diagnostics_(diag)
    {
        CELER_EXPECT(id_);
        CELER_EXPECT(diagnostics_);
    }

    //! Execute the action with host data
    void execute(CoreHostRef const& core) const final
    {
        this->execute_impl(core.states, diagnostics_->host);
    }

    //! Execute the action with device data
    void execute(CoreDeviceRef const& core) const final
    {
        this->execute_impl(core.states, diagnostics_->device);
    }

    //!@{
    //! Action interface
    ActionId    action_id() const final { return id_; }
    std::string label() const final { return "diagnostics"; }
    std::string description() const final
    {
        return "diagnostics after post-step";
    }
    ActionOrder order() const final { return ActionOrder::post_post; }
    //!@}

  private:
    ActionId      id_;
    SPDiagnostics diagnostics_;

    template<MemSpace M>
    using VecUPDiag = DiagnosticStore::VecUPDiag<M>;

    template<MemSpace M>
    void execute_impl(CoreStateData<Ownership::reference, M> const& states,
                      VecUPDiag<M> const& diagnostics) const
    {
        for (const auto& diag_ptr : diagnostics)
        {
            diag_ptr->mid_step(states);
        }
    }
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
//! Default virtual destructor
TransporterBase::~TransporterBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent problem data.
 */
template<MemSpace M>
Transporter<M>::Transporter(TransporterInput inp)
{
    TransporterBase::input_ = std::move(inp);
    CELER_EXPECT(input_);

    const CoreParams& params = *input_.params;

    // Create diagnostics
    if (input_.enable_diagnostics)
    {
        diagnostics_ = std::make_shared<DiagnosticStore>();
        auto& diag   = get_diag_ref(*diagnostics_, MemTag<M>{});
        diag.push_back(std::make_unique<StepDiagnostic<M>>(
            get_ref<M>(params), params.particle(), input_.num_track_slots, 200));
        diag.push_back(std::make_unique<ParticleProcessDiagnostic<M>>(
            get_ref<M>(params), params.particle(), params.physics()));
        {
            const auto& ediag = input_.energy_diag;
            CELER_VALIDATE(ediag.axis >= 'x' && ediag.axis <= 'z',
                           << "Invalid axis '" << ediag.axis
                           << "' (must be x, y, or z)");
            diag.push_back(std::make_unique<EnergyDiagnostic<M>>(
                linspace(ediag.min, ediag.max, ediag.num_bins + 1),
                static_cast<Axis>(ediag.axis - 'x')));
        }

        // Add diagnostic adapters to action manager
        diagnostic_action_ = params.action_reg()->next_id();
        params.action_reg()->insert(std::make_shared<DiagnosticActionAdapter>(
            diagnostic_action_, diagnostics_));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
TransporterResult Transporter<M>::operator()(VecPrimary primaries)
{
    Stopwatch get_transport_time;

    // Initialize results
    TransporterResult result;
    if (input_.max_steps != input_.no_max_steps())
    {
        result.time.steps.reserve(input_.max_steps);
        result.initializers.reserve(input_.max_steps);
        result.active.reserve(input_.max_steps);
        result.alive.reserve(input_.max_steps);
    }
    auto append_track_counts = [&result](const StepperResult& track_counts) {
        result.initializers.push_back(track_counts.queued);
        result.active.push_back(track_counts.active);
        result.alive.push_back(track_counts.alive);
    };

    // Abort cleanly for interrupt and user-defined signals
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};
    CELER_LOG(status) << "Transporting";

    StepperInput input;
    input.params             = input_.params;
    input.num_track_slots    = input_.num_track_slots;
    input.num_initializers   = input_.num_initializers;
    input.sync               = input_.sync;
    Stepper<M> step(std::move(input));

    Stopwatch get_step_time;
    size_type remaining_steps = input_.max_steps;

    // Copy primaries to device and transport the first step
    auto track_counts = step(std::move(primaries));
    append_track_counts(track_counts);
    result.time.steps.push_back(get_step_time());

    while (track_counts)
    {
        if (CELER_UNLIKELY(--remaining_steps == 0))
        {
            CELER_LOG(error) << "Exceeded step count of " << input_.max_steps
                             << ": aborting transport loop";
            break;
        }
        if (CELER_UNLIKELY(interrupted()))
        {
            CELER_LOG(error) << "Caught interrupt signal: aborting transport "
                                "loop";
            interrupted = {};
            break;
        }

        get_step_time = {};
        track_counts  = step();
        append_track_counts(track_counts);
        result.time.steps.push_back(get_step_time());
    }

    // Save kernel timing if host or synchronization is enabled
    if (M == MemSpace::host || input_.sync)
    {
        const auto& action_seq  = step.actions();
        const auto& action_ptrs = action_seq.actions();
        const auto& times       = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == times.size());
        for (auto i : range(action_ptrs.size()))
        {
            auto&& label               = action_ptrs[i]->label();
            result.time.actions[label] = times[i];
        }
    }

    if (diagnostics_)
    {
        CELER_LOG(status) << "Finalizing diagnostic data";
        // Collect results from diagnostics
        for (auto& diagnostic : get_diag_ref(*diagnostics_, MemTag<M>{}))
        {
            diagnostic->get_result(&result);
        }
    }
    result.time.total = get_transport_time();
    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Transporter<MemSpace::host>;
template class Transporter<MemSpace::device>;

//---------------------------------------------------------------------------//
} // namespace demo_loop
