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

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/VectorUtils.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/generated/BoundaryAction.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

#include "diagnostic/EnergyDiagnostic.hh"
#include "diagnostic/ParticleProcessDiagnostic.hh"
#include "diagnostic/StepDiagnostic.hh"
#include "diagnostic/TrackDiagnostic.hh"

using namespace demo_loop;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES AND FUNCTIONS
//---------------------------------------------------------------------------//
//!@{
//! Helpers for constructing parameters for host and device.
template<class P, MemSpace M>
struct ParamsGetter;

template<class P>
struct ParamsGetter<P, MemSpace::host>
{
    const P& params_;

    auto operator()() const -> decltype(auto) { return params_.host_ref(); }
};

template<class P>
struct ParamsGetter<P, MemSpace::device>
{
    const P& params_;

    auto operator()() const -> decltype(auto) { return params_.device_ref(); }
};

template<MemSpace M, class P>
decltype(auto) get_ref(const P& params)
{
    return ParamsGetter<P, M>{params}();
}

template<MemSpace M>
CoreParamsData<Ownership::const_reference, M>
build_params_refs(const TransporterInput& p, ActionId boundary_action)
{
    CELER_EXPECT(boundary_action);
    CoreParamsData<Ownership::const_reference, M> ref;
    ref.scalars.boundary_action        = boundary_action;
    ref.geometry                       = get_ref<M>(*p.geometry);
    ref.geo_mats                       = get_ref<M>(*p.geo_mats);
    ref.materials                      = get_ref<M>(*p.materials);
    ref.particles                      = get_ref<M>(*p.particles);
    ref.cutoffs                        = get_ref<M>(*p.cutoffs);
    ref.physics                        = get_ref<M>(*p.physics);
    ref.rng                            = get_ref<M>(*p.rng);
    CELER_ENSURE(ref);
    return ref;
}

//! Allow constructing StateCollection from params.
struct ParamsShim
{
    const TransporterInput& p;
    ActionId                boundary_action;

    CoreParamsData<Ownership::const_reference, MemSpace::host> host_ref() const
    {
        CELER_ASSERT(boundary_action);
        return build_params_refs<MemSpace::host>(p, boundary_action);
    }
};

//! Create a vector of diagnostics
template<MemSpace M>
std::vector<std::unique_ptr<Diagnostic<M>>>
build_diagnostics(const TransporterInput&                        inp,
                  CoreParamsData<Ownership::const_reference, M>& params)
{
    std::vector<std::unique_ptr<Diagnostic<M>>> result;
    if (inp.enable_diagnostics)
    {
        result.push_back(std::make_unique<TrackDiagnostic<M>>());
        result.push_back(std::make_unique<StepDiagnostic<M>>(
            params, inp.particles, inp.max_num_tracks, 200));
        result.push_back(std::make_unique<ParticleProcessDiagnostic<M>>(
            params, inp.particles, inp.physics));

        const auto& ediag = inp.energy_diag;
        CELER_VALIDATE(ediag.axis >= 'x' && ediag.axis <= 'z',
                       << "Invalid axis '" << ediag.axis
                       << "' (must be x, y, or z)");
        result.push_back(std::make_unique<EnergyDiagnostic<M>>(
            linspace(ediag.min, ediag.max, ediag.num_bins + 1),
            static_cast<Axis>(ediag.axis - 'x')));
    }
    return result;
}

//!@}
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
Transporter<M>::Transporter(TransporterInput inp) : input_(std::move(inp))
{
    CELER_EXPECT(input_);

    // Add geometry action
    boundary_action_ = input_.actions->next_id();
    input_.actions->insert(
        std::make_shared<celeritas::generated::BoundaryAction>(
            boundary_action_, "Geometry boundary"));

    // Build params
    params_ = build_params_refs<M>(input_, boundary_action_);
    CELER_ASSERT(params_);

    // TODO: add physics params accessors instead of looking these up as
    // strings
    pre_step_action_   = input_.actions->find_action("pre-step");
    along_step_action_ = input_.actions->find_action("along-step");
    discrete_select_action_
        = input_.actions->find_action("physics-discrete-select");
    CELER_ASSERT(pre_step_action_ && along_step_action_
                 && discrete_select_action_);

    // Create states
    states_ = CollectionStateStore<CoreStateData, M>(
        ParamsShim{input_, boundary_action_}, input_.max_num_tracks);
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
TransporterResult Transporter<M>::operator()(const TrackInitParams& primaries)
{
    CELER_LOG(status) << "Initializing primaries";
    Stopwatch get_transport_time;
    // Initialize results
    TransporterResult result;
    if (input_.max_steps != input_.no_max_steps())
    {
        result.time.steps.reserve(input_.max_steps);
        result.initializers.reserve(input_.max_steps);
        result.active.reserve(input_.max_steps);
    }

    // Construct diagnostics
    auto diagnostics = build_diagnostics(input_, params_);

    // Copy primaries to device and create track initializers.
    // We (currently) have to create initializers from *all* primaries
    // all at once.
    TrackInitStateData<Ownership::value, M> track_init_states;
    resize(&track_init_states, primaries.host_ref(), input_.max_num_tracks);
    CELER_ASSERT(primaries.host_ref().primaries.size()
                 <= track_init_states.initializers.capacity());
    extend_from_primaries(primaries.host_ref(), &track_init_states);

    // Create data manager
    CoreRef<M> core_ref;
    core_ref.params              = params_;
    core_ref.states              = states_.ref();
    const ActionManager& actions = *input_.actions;

    // Abort cleanly for interrupt and user-defined signals
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};
    CELER_LOG(status) << "Transporting";
    size_type num_alive       = 0;
    size_type num_inits       = track_init_states.initializers.size();
    size_type remaining_steps = input_.max_steps;

    while (num_alive > 0 || num_inits > 0)
    {
        // Start timers
        Stopwatch get_step_time;

        result.initializers.push_back(num_inits);

        // Create new tracks from primaries or secondaries
        initialize_tracks(core_ref, &track_init_states);
        result.active.push_back(input_.max_num_tracks
                                - track_init_states.vacancies.size());

        // Reset track data, sample mean free path, and calculate step limits
        actions.invoke<M>(pre_step_action_, core_ref);

        // Move, calculate dE/dx, and select model for discrete interaction
        actions.invoke<M>(along_step_action_, core_ref);

        // Cross boundary
        actions.invoke<M>(boundary_action_, core_ref);

        // Determine discrete processes
        actions.invoke<M>(discrete_select_action_, core_ref);

        // Launch the interaction kernels for all applicable models
        for (ActionId action : input_.physics->model_actions())
        {
            actions.invoke<M>(action, core_ref);
        }

        // Mid-step diagnostics
        for (auto& diagnostic : diagnostics)
        {
            diagnostic->mid_step(core_ref.states);
        }

        // Create track initializers from surviving secondaries
        extend_from_secondaries(core_ref, &track_init_states);

        // Get the number of track initializers and active tracks
        num_alive = input_.max_num_tracks - track_init_states.vacancies.size();
        num_inits = track_init_states.initializers.size();

        // End-of-step diagnostics
        for (auto& diagnostic : diagnostics)
        {
            diagnostic->end_step(core_ref.states);
        }

        result.time.steps.push_back(get_step_time());

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
    }

    CELER_LOG(status) << "Finalizing diagnostic data";
    // Collect results from diagnostics
    for (auto& diagnostic : diagnostics)
    {
        diagnostic->get_result(&result);
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
} // namespace celeritas
