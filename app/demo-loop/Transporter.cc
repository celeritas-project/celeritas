//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <csignal>
#include <memory>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/global/ActionRegistry.hh"  // IWYU pragma: keep
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/detail/ActionSequence.hh"
#include "celeritas/grid/VectorUtils.hh"
#include "celeritas/phys/Model.hh"

#include "diagnostic/Diagnostic.hh"
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
decltype(auto) get_diag_ref(DiagnosticStore& params)
{
    if constexpr (M == MemSpace::host)
    {
        return (params.host);
    }
    else if constexpr (M == MemSpace::device)
    {
        return (params.device);
    }
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
    void execute(ParamsHostCRef const& params, StateHostRef& states) const final
    {
        this->execute_impl(params, states, diagnostics_->host);
    }

    //! Execute the action with device data
    void
    execute(ParamsDeviceCRef const& params, StateDeviceRef& states) const final
    {
        this->execute_impl(params, states, diagnostics_->device);
    }

    //!@{
    //! \name Action interface
    ActionId action_id() const final { return id_; }
    std::string label() const final { return "diagnostics"; }
    std::string description() const final
    {
        return "diagnostics after post-step";
    }
    ActionOrder order() const final { return ActionOrder::post_post; }
    //!@}

  private:
    ActionId id_;
    SPDiagnostics diagnostics_;

    template<MemSpace M>
    using VecUPDiag = DiagnosticStore::VecUPDiag<M>;

    template<MemSpace M>
    void
    execute_impl(CoreParamsData<Ownership::const_reference, M> const& params,
                 CoreStateData<Ownership::reference, M> const& states,
                 VecUPDiag<M> const& diagnostics) const
    {
        for (auto const& diag_ptr : diagnostics)
        {
            diag_ptr->mid_step(params, states);
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
//! Default virtual destructor
TransporterBase::~TransporterBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent problem data.
 */
template<MemSpace M>
Transporter<M>::Transporter(TransporterInput inp) : max_steps_(inp.max_steps)
{
    CELER_EXPECT(inp);

    CoreParams const& params = *inp.params;

    // Create diagnostics
    // TODO: these should be actions with StreamStores
    if (inp.enable_diagnostics)
    {
        diagnostics_ = std::make_shared<DiagnosticStore>();
        auto& diag = get_diag_ref<M>(*diagnostics_);
        diag.push_back(std::make_unique<StepDiagnostic<M>>(
            get_ref<M>(params), params.particle(), inp.num_track_slots, 200));

        // Add diagnostic adapters to action manager
        diagnostic_action_ = params.action_reg()->next_id();
        params.action_reg()->insert(std::make_shared<DiagnosticActionAdapter>(
            diagnostic_action_, diagnostics_));
    }

    // Create stepper
    StepperInput step_input;
    step_input.params = inp.params;
    step_input.num_track_slots = inp.num_track_slots;
    step_input.stream_id = inp.stream_id;
    step_input.sync = inp.sync;
    stepper_ = std::make_shared<Stepper<M>>(std::move(step_input));
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
RunnerResult Transporter<M>::operator()(SpanConstPrimary primaries)
{
    Stopwatch get_transport_time;

    // Initialize results
    TransporterResult result;
    auto append_track_counts = [&result](StepperResult const& track_counts) {
        result.initializers.push_back(track_counts.queued);
        result.active.push_back(track_counts.active);
        result.alive.push_back(track_counts.alive);
    };

    // Abort cleanly for interrupt and user-defined signals
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};
    CELER_LOG(status) << "Transporting";

    Stopwatch get_step_time;
    size_type remaining_steps = max_steps_;

    auto& step = *stepper_;
    // Copy primaries to device and transport the first step
    auto track_counts = step(primaries);
    append_track_counts(track_counts);
    result.time.steps.push_back(get_step_time());

    while (track_counts)
    {
        if (CELER_UNLIKELY(--remaining_steps == 0))
        {
            CELER_LOG(error) << "Exceeded step count of " << max_steps_
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
        track_counts = step();
        append_track_counts(track_counts);
        result.time.steps.push_back(get_step_time());
    }

    // Save kernel timing if host or synchronization is enabled
    auto const& action_seq = step.actions();
    if (M == MemSpace::host || action_seq.sync())
    {
        auto const& action_ptrs = action_seq.actions();
        auto const& times = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == times.size());
        for (auto i : range(action_ptrs.size()))
        {
            auto&& label = action_ptrs[i]->label();
            result.time.actions[label] = times[i];
        }
    }

    if (diagnostics_)
    {
        CELER_LOG(status) << "Finalizing diagnostic data";
        // Collect results from diagnostics
        for (auto& diagnostic : get_diag_ref<M>(*diagnostics_))
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
}  // namespace demo_loop
