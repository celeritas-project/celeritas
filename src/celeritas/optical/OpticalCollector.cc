//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ActionRegistryOutput.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"
#include "CoreState.hh"
#include "MaterialParams.hh"
#include "PhysicsParams.hh"
#include "TrackInitParams.hh"
#include "gen/CherenkovParams.hh"
#include "gen/ScintillationParams.hh"
#include "gen/detail/GeneratorAction.hh"
#include "gen/detail/OffloadAction.hh"
#include "gen/detail/OffloadGatherAction.hh"

#include "detail/OpticalLaunchAction.hh"
#include "detail/OpticalSizes.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with core data and optical data.
 *
 * This adds several actions and auxiliary data to the registry.
 */
OpticalCollector::OpticalCollector(CoreParams const& core, Input&& inp)
{
    CELER_EXPECT(inp);

    // Create action to gather pre-step data for populating distributions
    gather_ = detail::OffloadGatherAction::make_and_insert(core);

    // The offload, generator, and launch actions much be created in a specific
    // order but require auxiliary data IDs from actions created later.
    // Precalculate the IDs for the generator and optical state aux data.
    size_type num_gen = !!inp.cherenkov + !!inp.scintillation;
    auto gen_aux_id = core.aux_reg()->next_id();
    auto optical_aux_id = core.aux_reg()->next_id() + num_gen;

    if (inp.cherenkov)
    {
        // Create action to generate Cherenkov optical distributions
        OffloadAction<GT::cherenkov>::Input oa_inp;
        oa_inp.step_id = gather_->aux_id();
        oa_inp.gen_id = gen_aux_id++;
        oa_inp.optical_id = optical_aux_id;
        oa_inp.material = inp.material;
        oa_inp.shared = inp.cherenkov;
        cherenkov_offload_ = OffloadAction<GT::cherenkov>::make_and_insert(
            core, std::move(oa_inp));
    }
    if (inp.scintillation)
    {
        // Create action to generate scintillation optical distributions
        OffloadAction<GT::scintillation>::Input oa_inp;
        oa_inp.step_id = gather_->aux_id();
        oa_inp.gen_id = gen_aux_id++;
        oa_inp.optical_id = optical_aux_id;
        oa_inp.material = inp.material;
        oa_inp.shared = inp.scintillation;
        scint_offload_ = OffloadAction<GT::scintillation>::make_and_insert(
            core, std::move(oa_inp));
    }

    // Create optical core params
    auto optical_params = std::make_shared<optical::CoreParams>([&] {
        optical::CoreParams::Input op_inp;
        op_inp.geometry = core.geometry();
        op_inp.material = inp.material;
        // TODO: unique RNG streams for optical loop
        op_inp.rng = core.rng();
        op_inp.init = std::make_shared<optical::TrackInitParams>(
            inp.initializer_capacity);
        op_inp.surface = core.surface();
        op_inp.action_reg = std::make_shared<ActionRegistry>();
        op_inp.max_streams = core.max_streams();
        {
            optical::PhysicsParams::Input pp_inp;
            pp_inp.model_builders = std::move(inp.model_builders);
            pp_inp.materials = op_inp.material;
            pp_inp.action_registry = op_inp.action_reg.get();
            op_inp.physics
                = std::make_shared<optical::PhysicsParams>(std::move(pp_inp));
        }
        op_inp.detector_labels = inp.detector_labels;
        CELER_ENSURE(op_inp);
        return op_inp;
    }());

    if (inp.cherenkov)
    {
        // Create action to generate Cherenkov primaries
        GeneratorAction<GT::cherenkov>::Input ga_inp;
        ga_inp.optical_id = optical_aux_id;
        ga_inp.material = inp.material;
        ga_inp.shared = inp.cherenkov;
        ga_inp.capacity = inp.buffer_capacity;
        cherenkov_generate_ = GeneratorAction<GT::cherenkov>::make_and_insert(
            core, *optical_params, std::move(ga_inp));
    }
    if (inp.scintillation)
    {
        // Create action to generate scintillation primaries
        GeneratorAction<GT::scintillation>::Input ga_inp;
        ga_inp.optical_id = optical_aux_id;
        ga_inp.material = inp.material;
        ga_inp.shared = inp.scintillation;
        ga_inp.capacity = inp.buffer_capacity;
        scint_generate_ = GeneratorAction<GT::scintillation>::make_and_insert(
            core, *optical_params, std::move(ga_inp));
    }

    // Save optical diagnostic information
    core.output_reg()->insert(std::make_shared<ActionRegistryOutput>(
        optical_params->action_reg(), "optical-actions"));

    // Create launch action with optical params+state and access to gen data
    detail::OpticalLaunchAction::Input la_inp;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.max_step_iters = inp.max_step_iters;
    la_inp.auto_flush = inp.auto_flush;
    la_inp.optical_params = std::move(optical_params);
    launch_ = detail::OpticalLaunchAction::make_and_insert(core,
                                                           std::move(la_inp));

    // Add optical sizes
    detail::OpticalSizes sizes;
    sizes.streams = core.max_streams();
    sizes.generators = sizes.streams * inp.buffer_capacity;
    sizes.initializers = sizes.streams * inp.initializer_capacity;
    sizes.tracks = sizes.streams * inp.num_track_slots;

    core.output_reg()->insert(
        OutputInterfaceAdapter<detail::OpticalSizes>::from_rvalue_ref(
            OutputInterface::Category::internal,
            "optical-sizes",
            std::move(sizes)));

    // Launch action must be *after* offload and generator actions
    CELER_ENSURE(!cherenkov_offload_
                 || launch_->action_id() > cherenkov_offload_->action_id());
    CELER_ENSURE(!scint_offload_
                 || launch_->action_id() > scint_offload_->action_id());
    CELER_ENSURE(!cherenkov_generate_
                 || launch_->action_id() > cherenkov_generate_->action_id());
    CELER_ENSURE(!scint_generate_
                 || launch_->action_id() > scint_generate_->action_id());
    CELER_ENSURE(this->optical_aux_id() == optical_aux_id);
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical core state data.
 */
AuxId OpticalCollector::optical_aux_id() const
{
    return launch_->aux_id();
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical Cherenkov offload data.
 */
AuxId OpticalCollector::cherenkov_aux_id() const
{
    return cherenkov_generate_ ? cherenkov_generate_->aux_id() : AuxId{};
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical scintillation offload data.
 */
AuxId OpticalCollector::scintillation_aux_id() const
{
    return scint_generate_ ? scint_generate_->aux_id() : AuxId{};
}

//---------------------------------------------------------------------------//
/*!
 * Get and reset cumulative statistics on optical generation from a state.
 */
OpticalAccumStats OpticalCollector::exchange_counters(AuxStateVec& aux) const
{
    auto& state = dynamic_cast<optical::CoreStateBase&>(
        aux.at(this->optical_aux_id()));
    auto& accum = state.accum();

    if (auto id = this->cherenkov_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        accum.cherenkov = gen.accum;
    }
    if (auto id = this->scintillation_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        accum.scintillation = gen.accum;
    }

    return std::exchange(accum, {});
}

//---------------------------------------------------------------------------//
/*!
 * Get info on the number of tracks in the buffer.
 */
auto OpticalCollector::buffer_counts(AuxStateVec const& aux) const
    -> OpticalBufferSize
{
    OpticalBufferSize result;

    auto const& state = dynamic_cast<optical::CoreStateBase const&>(
        aux.at(this->optical_aux_id()));
    result.photons = state.counters().num_pending;

    if (auto id = this->cherenkov_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        result.distributions += gen.buffer_size;
    }
    if (auto id = this->scintillation_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        result.distributions += gen.buffer_size;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
