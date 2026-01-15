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
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "CoreState.hh"
#include "MaterialParams.hh"
#include "PhysicsParams.hh"
#include "gen/CherenkovParams.hh"
#include "gen/GeneratorAction.hh"
#include "gen/OffloadAction.hh"
#include "gen/OffloadGatherAction.hh"
#include "gen/ScintillationParams.hh"

#include "detail/OpticalLaunchAction.hh"

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

    if (inp.action_times)
    {
        // Create aux data to accumulate optical action times
        action_times_
            = ActionTimes::make_and_insert(inp.optical_params->action_reg(),
                                           core.aux_reg(),
                                           "optial-action-times");
    }

    // Create launch action with optical params+state and access to aux data
    detail::OpticalLaunchAction::Input la_inp;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.auto_flush = inp.auto_flush;
    la_inp.action_times = action_times_;
    la_inp.optical_params = inp.optical_params;
    launch_ = detail::OpticalLaunchAction::make_and_insert(core,
                                                           std::move(la_inp));

    // Create core action to gather pre-step data for populating distributions
    pre_gather_
        = OffloadGatherAction<StepActionOrder::pre>::make_and_insert(core);

    // Create optical action to generate Cherenkov or scintillation photons
    generate_ = optical::GeneratorAction::make_and_insert(*inp.optical_params,
                                                          inp.buffer_capacity);

    if (inp.optical_params->cherenkov())
    {
        // Create core action to generate Cherenkov optical distributions
        OffloadAction<GT::cherenkov>::Input oa_inp;
        oa_inp.pre_step_id = pre_gather_->aux_id();
        oa_inp.gen_id = generate_->aux_id();
        oa_inp.optical_id = launch_->aux_id();
        oa_inp.material = inp.optical_params->material();
        oa_inp.shared = inp.optical_params->cherenkov();
        cherenkov_offload_ = OffloadAction<GT::cherenkov>::make_and_insert(
            core, std::move(oa_inp));
    }
    if (inp.optical_params->scintillation())
    {
        // Create core action to gather post-along-step state data
        pre_post_gather_
            = OffloadGatherAction<StepActionOrder::pre_post>::make_and_insert(
                core);

        // Create action to generate scintillation optical distributions
        OffloadAction<GT::scintillation>::Input oa_inp;
        oa_inp.pre_step_id = pre_gather_->aux_id();
        oa_inp.pre_post_step_id = pre_post_gather_->aux_id();
        oa_inp.gen_id = generate_->aux_id();
        oa_inp.optical_id = launch_->aux_id();
        oa_inp.material = inp.optical_params->material();
        oa_inp.shared = inp.optical_params->scintillation();
        scint_offload_ = OffloadAction<GT::scintillation>::make_and_insert(
            core, std::move(oa_inp));
    }

    // Save core params
    optical_params_ = std::move(inp.optical_params);

    CELER_ENSURE(launch_);
    CELER_ENSURE(optical_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Access optical state.
 */
optical::CoreStateBase const&
OpticalCollector::optical_state(CoreStateInterface const& core) const
{
    auto& state = dynamic_cast<optical::CoreStateBase const&>(
        core.aux().at(launch_->aux_id()));
    return state;
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
/*!
 * Get the accumulated action times.
 */
auto OpticalCollector::get_action_times(AuxStateVec const& aux) const
    -> MapStrDbl
{
    if (action_times_)
    {
        return action_times_->get_action_times(aux);
    }
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
