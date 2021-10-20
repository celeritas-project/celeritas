//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoRun.i.hh
//---------------------------------------------------------------------------//
#include "base/CollectionStateStore.hh"
#include "comm/Logger.hh"
#include "physics/base/ModelInterface.hh"
#include "sim/TrackInitUtils.hh"
#include "sim/TrackInterface.hh"
#include "LDemoParams.hh"
#include "LDemoKernel.hh"
#include "ParticleProcessDiagnostic.hh"
#include "TrackDiagnostic.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
template<class P, MemSpace M>
struct ParamsGetter;

template<class P>
struct ParamsGetter<P, MemSpace::host>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.host_pointers();
    }
};

template<class P>
struct ParamsGetter<P, MemSpace::device>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.device_pointers();
    }
};

template<MemSpace M, class P>
decltype(auto) get_pointers(const P& params)
{
    return ParamsGetter<P, M>{params}();
}

//---------------------------------------------------------------------------//
template<MemSpace M>
ParamsData<Ownership::const_reference, M>
build_params_refs(const LDemoParams& p)
{
    ParamsData<Ownership::const_reference, M> ref;
    ref.geometry    = get_pointers<M>(*p.geometry);
    ref.materials   = get_pointers<M>(*p.materials);
    ref.geo_mats    = get_pointers<M>(*p.geo_mats);
    ref.cutoffs     = get_pointers<M>(*p.cutoffs);
    ref.particles   = get_pointers<M>(*p.particles);
    ref.physics     = get_pointers<M>(*p.physics);
    ref.rng         = get_pointers<M>(*p.rng);
    ref.track_inits = get_pointers<M>(*p.track_inits);
    CELER_ENSURE(ref);
    return ref;
}

//---------------------------------------------------------------------------//
/*!
 * Launch interaction kernels for all applicable models.
 *
 * For now, just launch *all* the models.
 */
template<MemSpace M>
void launch_models(LDemoParams const& host_params,
                   ParamsData<Ownership::const_reference, M> const& params,
                   StateData<Ownership::reference, M> const&        states)
{
    // TODO: these *should* be able to be persistent across steps, rather than
    // recreated at every step.
    ModelInteractRefs<M> refs;
    refs.params.particle     = params.particles;
    refs.params.material     = params.materials;
    refs.params.physics      = params.physics;
    refs.params.cutoffs      = params.cutoffs;
    refs.states.particle     = states.particles;
    refs.states.material     = states.materials;
    refs.states.physics      = states.physics;
    refs.states.rng          = states.rng;
    refs.states.sim          = states.sim;
    refs.states.direction    = states.geometry.dir;
    refs.states.secondaries  = states.secondaries;
    refs.states.interactions = states.interactions;
    CELER_ASSERT(refs);

    // Loop over physics models IDs and invoke `interact`
    for (auto model_id : range(ModelId{host_params.physics->num_models()}))
    {
        const Model& model = host_params.physics->model(model_id);
        model.interact(refs);
    }
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
template<MemSpace M>
LDemoResult run_demo(LDemoArgs args)
{
    CELER_EXPECT(args);

    // Load all the problem data
    LDemoParams params = load_params(args);

    // Create param interfaces
    auto params_ref = build_params_refs<M>(params);

    // Diagnostics
    // TODO: Create a vector of these objects.
    TrackDiagnostic<M>           track_diagnostic;
    ParticleProcessDiagnostic<M> process_diagnostic(
        params_ref, params.particles, params.physics);

    // Create states (TODO state store?)
    StateData<Ownership::value, M> state_storage;
    resize(&state_storage,
           build_params_refs<MemSpace::host>(params),
           args.max_num_tracks);
    StateData<Ownership::reference, M> states_ref = make_ref(state_storage);

    // Copy primaries to device and create track initializers
    // TODO: for now this assumes we can initialize all primaries at once, but
    // we should also handle the case where we have more primaries than tracks
    CELER_ASSERT(params.track_inits->host_pointers().primaries.size()
                 <= state_storage.track_inits.initializers.capacity());
    extend_from_primaries(params.track_inits->host_pointers(),
                          &state_storage.track_inits);

    size_type num_alive       = 0;
    size_type num_inits       = state_storage.track_inits.initializers.size();
    size_type remaining_steps = args.max_steps;

    while (num_alive > 0 || num_inits > 0)
    {
        // Create new tracks from primaries or secondaries
        initialize_tracks(params_ref, states_ref, &state_storage.track_inits);

        demo_loop::pre_step(params_ref, states_ref);
        demo_loop::along_and_post_step(params_ref, states_ref);

        // Mid-step diagnostics
        process_diagnostic.mid_step(states_ref);

        // Launch the interaction kernels for all applicable models
        launch_models(params, params_ref, states_ref);

        // Postprocess secondaries and interaction results
        demo_loop::process_interactions(params_ref, states_ref);

        // Create track initializers from surviving secondaries
        extend_from_secondaries(
            params_ref, states_ref, &state_storage.track_inits);

        // Clear secondaries
        demo_loop::cleanup(params_ref, states_ref);

        // Get the number of track initializers and active tracks
        num_alive = args.max_num_tracks
                    - state_storage.track_inits.vacancies.size();
        num_inits = state_storage.track_inits.initializers.size();

        // End-of-step diagnostic(s)
        track_diagnostic.end_step(states_ref);

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // TODO: return result
    return LDemoResult{{0},
                       track_diagnostic.num_alive_per_step(),
                       {0},
                       process_diagnostic.particle_processes(),
                       0};
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
