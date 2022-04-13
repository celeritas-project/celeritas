//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <csignal>
#include <memory>

#include "base/Assert.hh"
#include "base/Stopwatch.hh"
#include "base/VectorUtils.hh"
#include "comm/Logger.hh"
#include "comm/ScopedSignalHandler.hh"
#include "geometry/GeoMaterialParams.hh"
#include "geometry/GeoParams.hh"
#include "geometry/generated/BoundaryAction.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/em/AtomicRelaxationParams.hh"
#include "physics/material/MaterialParams.hh"
#include "random/RngParams.hh"
#include "sim/ActionManager.hh"
#include "sim/TrackInitParams.hh"
#include "sim/TrackInitUtils.hh"

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
    ref.scalars.boundary_action = boundary_action;
    ref.scalars.secondary_stack_factor = p.secondary_stack_factor;
    ref.geometry                       = get_ref<M>(*p.geometry);
    ref.geo_mats                       = get_ref<M>(*p.geo_mats);
    ref.materials                      = get_ref<M>(*p.materials);
    ref.particles                      = get_ref<M>(*p.particles);
    ref.cutoffs                        = get_ref<M>(*p.cutoffs);
    ref.physics                        = get_ref<M>(*p.physics);
    ref.rng                            = get_ref<M>(*p.rng);
    if (p.relaxation)
    {
        ref.relaxation = get_ref<M>(*p.relaxation);
    }
    CELER_ENSURE(ref);
    return ref;
}

//! Allow constructing StateCollection from params.
struct ParamsShim
{
    const TransporterInput& p;
    ActionId boundary_action;

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

// Accumulate fine-grained timing results
template<MemSpace M>
void accum_time(const TransporterInput&, Stopwatch&, real_type*);

template<>
void accum_time<MemSpace::device>(const TransporterInput& inp,
                                  Stopwatch&              get_time,
                                  real_type*              time)
{
    if (inp.sync)
    {
        CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
        *time += get_time();
    }
}

template<>
void accum_time<MemSpace::host>(const TransporterInput&,
                                Stopwatch& get_time,
                                real_type* time)
{
    *time += get_time();
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
    pre_step_action_ = input_.actions->find_action("pre-step");
    along_step_action_      = input_.actions->find_action("along-step");
    discrete_select_action_ = input_.actions->find_action("physics-discrete-select");
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
    core_ref.params = params_;
    core_ref.states = states_.ref();
    const ActionManager& actions = *input_.actions;

    ScopedSignalHandler interrupted(SIGINT);
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
        Stopwatch get_time;
        initialize_tracks(core_ref, &track_init_states);
        accum_time<M>(input_, get_time, &result.time.initialize_tracks);

        result.active.push_back(input_.max_num_tracks
                                - track_init_states.vacancies.size());

        // Reset track data, sample mean free path, and calculate step limits
        get_time = {};
        actions.invoke<M>(pre_step_action_, core_ref);
        accum_time<M>(input_, get_time, &result.time.pre_step);

        // Move, calculate dE/dx, and select model for discrete interaction
        get_time = {};
        actions.invoke<M>(along_step_action_, core_ref);
        accum_time<M>(input_, get_time, &result.time.along_step);

        // Cross boundary
        get_time = {};
        actions.invoke<M>(boundary_action_, core_ref);
        accum_time<M>(input_, get_time, &result.time.cross_boundary);

        // Determine discrete processes
        get_time = {};
        actions.invoke<M>(discrete_select_action_, core_ref);
        accum_time<M>(input_, get_time, &result.time.discrete_select);

        // Launch the interaction kernels for all applicable models
        get_time = {};
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
        get_time = {};
        extend_from_secondaries(core_ref, &track_init_states);
        accum_time<M>(input_, get_time, &result.time.extend_from_secondaries);

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
