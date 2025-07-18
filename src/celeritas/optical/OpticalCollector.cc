//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ActionRegistryOutput.hh"
#include "celeritas/global/CoreParams.hh"

#include "CoreParams.hh"
#include "CoreState.hh"
#include "MaterialParams.hh"
#include "PhysicsParams.hh"
#include "gen/CherenkovParams.hh"
#include "gen/GeneratorAction.hh"
#include "gen/OffloadAction.hh"
#include "gen/OffloadGatherAction.hh"
#include "gen/ScintillationParams.hh"

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

    // Create optical core params
    auto optical_params = std::make_shared<optical::CoreParams>([&] {
        optical::CoreParams::Input op_inp;
        op_inp.geometry = core.geometry();
        op_inp.material = inp.material;
        // TODO: unique RNG streams for optical loop
        op_inp.rng = core.rng();
        op_inp.surface = core.surface();
        op_inp.action_reg = std::make_shared<ActionRegistry>();
        op_inp.gen_reg = std::make_shared<GeneratorRegistry>();
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

    // Create launch action with optical params+state and access to aux data
    detail::OpticalLaunchAction::Input la_inp;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.max_step_iters = inp.max_step_iters;
    la_inp.auto_flush = inp.auto_flush;
    la_inp.optical_params = optical_params;
    launch_ = detail::OpticalLaunchAction::make_and_insert(core,
                                                           std::move(la_inp));

    // Create core action to gather pre-step data for populating distributions
    gather_ = OffloadGatherAction::make_and_insert(core);

    if (inp.cherenkov)
    {
        // Create optical action to generate Cherenkov primaries
        optical::GeneratorAction<GT::cherenkov>::Input ga_inp;
        ga_inp.material = inp.material;
        ga_inp.shared = inp.cherenkov;
        ga_inp.capacity = inp.buffer_capacity;
        cherenkov_generate_
            = optical::GeneratorAction<GT::cherenkov>::make_and_insert(
                core, *optical_params, std::move(ga_inp));

        // Create core action to generate Cherenkov optical distributions
        OffloadAction<GT::cherenkov>::Input oa_inp;
        oa_inp.step_id = gather_->aux_id();
        oa_inp.gen_id = cherenkov_generate_->aux_id();
        oa_inp.optical_id = launch_->aux_id();
        oa_inp.material = inp.material;
        oa_inp.shared = inp.cherenkov;
        cherenkov_offload_ = OffloadAction<GT::cherenkov>::make_and_insert(
            core, std::move(oa_inp));
    }
    if (inp.scintillation)
    {
        // Create action to generate scintillation primaries
        optical::GeneratorAction<GT::scintillation>::Input ga_inp;
        ga_inp.material = inp.material;
        ga_inp.shared = inp.scintillation;
        ga_inp.capacity = inp.buffer_capacity;
        scint_generate_
            = optical::GeneratorAction<GT::scintillation>::make_and_insert(
                core, *optical_params, std::move(ga_inp));

        // Create action to generate scintillation optical distributions
        OffloadAction<GT::scintillation>::Input oa_inp;
        oa_inp.step_id = gather_->aux_id();
        oa_inp.gen_id = scint_generate_->aux_id();
        oa_inp.optical_id = launch_->aux_id();
        oa_inp.material = inp.material;
        oa_inp.shared = inp.scintillation;
        scint_offload_ = OffloadAction<GT::scintillation>::make_and_insert(
            core, std::move(oa_inp));
    }

    // Save optical diagnostic information
    core.output_reg()->insert(std::make_shared<ActionRegistryOutput>(
        optical_params->action_reg(), "optical-actions"));

    // Add optical sizes
    detail::OpticalSizes sizes;
    sizes.streams = core.max_streams();
    sizes.generators = sizes.streams * inp.buffer_capacity;
    sizes.tracks = sizes.streams * inp.num_track_slots;

    core.output_reg()->insert(
        OutputInterfaceAdapter<detail::OpticalSizes>::from_rvalue_ref(
            OutputInterface::Category::internal,
            "optical-sizes",
            std::move(sizes)));

    // Save core params
    optical_params_ = optical_params;

    CELER_ENSURE(optical_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Access Cherenkov params (may be null).
 */
auto OpticalCollector::cherenkov() const -> SPConstCherenkov
{
    if (!cherenkov_offload_)
        return nullptr;
    return cherenkov_offload_->params();
}

//---------------------------------------------------------------------------//
/*!
 * Access scintillation params (may be null).
 */
auto OpticalCollector::scintillation() const -> SPConstScintillation
{
    if (!scint_offload_)
        return nullptr;
    return scint_offload_->params();
}

//---------------------------------------------------------------------------//
/*!
 * Get the generator registry.
 */
GeneratorRegistry const& OpticalCollector::gen_reg() const
{
    return *optical_params_->gen_reg();
}

//---------------------------------------------------------------------------//
/*!
 * Get and reset cumulative statistics on optical generation from a state.
 */
OpticalAccumStats OpticalCollector::exchange_counters(AuxStateVec& aux) const
{
    auto& state
        = dynamic_cast<optical::CoreStateBase&>(aux.at(launch_->aux_id()));
    auto& accum = state.accum();

    accum.generators.resize(this->gen_reg().size());
    for (auto id : range(GeneratorId{this->gen_reg().size()}))
    {
        auto& gen_accum = this->gen_reg().at(id)->counters(aux).accum;
        accum.generators[id.get()] = std::exchange(gen_accum, {});
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

    for (auto id : range(GeneratorId{this->gen_reg().size()}))
    {
        auto const& counters = this->gen_reg().at(id)->counters(aux).counters;
        result.buffer_size += counters.buffer_size;
        result.num_pending += counters.num_pending;
        result.num_generated += counters.num_generated;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
