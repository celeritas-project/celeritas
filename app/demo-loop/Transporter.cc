//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include "base/VectorUtils.hh"
#include "geometry/GeoMaterialParams.hh"
#include "geometry/GeoParams.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/material/MaterialParams.hh"
#include "random/RngParams.hh"
#include "sim/TrackInitParams.hh"
#include "sim/TrackInitUtils.hh"

// Local includes for now
#include "diagnostic/EnergyDiagnostic.hh"
#include "diagnostic/ParticleProcessDiagnostic.hh"
#include "diagnostic/StepDiagnostic.hh"
#include "diagnostic/TrackDiagnostic.hh"
#include "LDemoKernel.hh"

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
ParamsData<Ownership::const_reference, M>
build_params_refs(const TransporterInput& p)
{
    ParamsData<Ownership::const_reference, M> ref;
    ref.control.secondary_stack_factor = p.secondary_stack_factor;
    ref.geometry                       = get_ref<M>(*p.geometry);
    ref.materials                      = get_ref<M>(*p.materials);
    ref.geo_mats                       = get_ref<M>(*p.geo_mats);
    ref.cutoffs                        = get_ref<M>(*p.cutoffs);
    ref.particles                      = get_ref<M>(*p.particles);
    ref.physics                        = get_ref<M>(*p.physics);
    ref.rng                            = get_ref<M>(*p.rng);
    CELER_ENSURE(ref);
    return ref;
}

//! Allow constructing StateCollection from params.
struct ParamsShim
{
    const TransporterInput& p;

    ParamsData<Ownership::const_reference, MemSpace::host> host_ref() const
    {
        return build_params_refs<MemSpace::host>(p);
    }
};

//!@}
//---------------------------------------------------------------------------//
/*!
 * Launch interaction kernels for all applicable models.
 *
 * For now, just launch *all* the models.
 */
template<MemSpace M>
void launch_models(TransporterInput const& host_params,
                   ParamsData<Ownership::const_reference, M> const& params,
                   StateData<Ownership::reference, M> const&        states)
{
    // TODO: these *should* be able to be persistent across steps, rather than
    // recreated at every step.
    ModelInteractRef<M> refs;
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
    params_ = build_params_refs<M>(input_);
    CELER_ASSERT(params_);
    states_ = CollectionStateStore<StateData, M>(ParamsShim{input_},
                                                 input_.max_num_tracks);
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
TransporterResult Transporter<M>::operator()(const TrackInitParams& primaries)
{
    // Diagnostics
    // TODO: Create a vector of these objects.
    TrackDiagnostic<M> track_diagnostic;
    StepDiagnostic<M>  step_diagnostic(
        params_, input_.particles, input_.max_num_tracks, 200);
    ParticleProcessDiagnostic<M> process_diagnostic(
        params_, input_.particles, input_.physics);
    EnergyDiagnostic<M> energy_diagnostic(linspace(-700.0, 700.0, 1024 + 1));

    // Copy primaries to device and create track initializers
    TrackInitStateData<Ownership::value, M> track_init_states;
    resize(&track_init_states, primaries.host_ref(), input_.max_num_tracks);
    CELER_ASSERT(primaries.host_ref().primaries.size()
                 <= track_init_states.initializers.capacity());
    extend_from_primaries(primaries.host_ref(), &track_init_states);

    size_type num_alive       = 0;
    size_type num_inits       = track_init_states.initializers.size();
    size_type remaining_steps = input_.max_steps;

    while (num_alive > 0 || num_inits > 0)
    {
        // Create new tracks from primaries or secondaries
        initialize_tracks(params_, states_.ref(), &track_init_states);

        demo_loop::pre_step(params_, states_.ref());
        demo_loop::along_and_post_step(params_, states_.ref());

        // Launch the interaction kernels for all applicable models
        launch_models(input_, params_, states_.ref());

        // Mid-step diagnostics
        process_diagnostic.mid_step(states_.ref());
        step_diagnostic.mid_step(states_.ref());

        // Postprocess secondaries and interaction results
        demo_loop::process_interactions(params_, states_.ref());

        // Create track initializers from surviving secondaries
        extend_from_secondaries(params_, states_.ref(), &track_init_states);

        // Clear secondaries
        demo_loop::cleanup(params_, states_.ref());

        // Get the number of track initializers and active tracks
        num_alive = input_.max_num_tracks - track_init_states.vacancies.size();
        num_inits = track_init_states.initializers.size();

        // End-of-step diagnostic(s)
        track_diagnostic.end_step(states_.ref());
        energy_diagnostic.end_step(states_.ref());

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // Collect results from diagnostics
    TransporterResult result;
    result.time       = {0};
    result.alive      = track_diagnostic.num_alive_per_step();
    result.edep       = energy_diagnostic.energy_deposition();
    result.process    = process_diagnostic.particle_processes();
    result.steps      = step_diagnostic.steps();
    result.total_time = 0;
    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Transporter<MemSpace::host>;
template class Transporter<MemSpace::device>;

//---------------------------------------------------------------------------//
} // namespace celeritas
