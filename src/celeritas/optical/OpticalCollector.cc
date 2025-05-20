//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/gen/OffloadParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"
#include "MaterialParams.hh"
#include "gen/CherenkovParams.hh"
#include "gen/OffloadData.hh"
#include "gen/ScintillationParams.hh"
#include "gen/detail/CherenkovGeneratorAction.hh"
#include "gen/detail/CherenkovOffloadAction.hh"
#include "gen/detail/OffloadGatherAction.hh"
#include "gen/detail/ScintGeneratorAction.hh"
#include "gen/detail/ScintOffloadAction.hh"

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

    OffloadOptions setup;
    setup.cherenkov = inp.cherenkov && inp.material;
    setup.scintillation = static_cast<bool>(inp.scintillation);
    setup.capacity = inp.buffer_capacity;

    // Create offload params
    AuxParamsRegistry& aux = *core.aux_reg();
    offload_params_ = std::make_shared<OffloadParams>(aux.next_id(), setup);
    aux.insert(offload_params_);

    // Action to gather pre-step data needed to generate optical distributions
    ActionRegistry& actions = *core.action_reg();
    gather_action_ = std::make_shared<detail::OffloadGatherAction>(
        actions.next_id(), offload_params_->aux_id());
    actions.insert(gather_action_);

    if (setup.cherenkov)
    {
        // Action to generate Cherenkov optical distributions
        cherenkov_action_ = std::make_shared<detail::CherenkovOffloadAction>(
            actions.next_id(),
            offload_params_->aux_id(),
            inp.material,
            inp.cherenkov);
        actions.insert(cherenkov_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_action_ = std::make_shared<detail::ScintOffloadAction>(
            actions.next_id(), offload_params_->aux_id(), inp.scintillation);
        actions.insert(scint_action_);
    }

    if (setup.cherenkov)
    {
        // Action to generate Cherenkov primaries
        cherenkov_gen_action_
            = std::make_shared<detail::CherenkovGeneratorAction>(
                actions.next_id(),
                offload_params_->aux_id(),
                // TODO: Hack: generator action must be before launch action
                // but needs optical state aux ID
                core.aux_reg()->next_id(),
                inp.material,
                std::move(inp.cherenkov),
                inp.auto_flush);
        actions.insert(cherenkov_gen_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation primaries
        scint_gen_action_ = std::make_shared<detail::ScintGeneratorAction>(
            actions.next_id(),
            offload_params_->aux_id(),
            // TODO: Hack: generator action must be before launch action
            // but needs optical state aux ID
            core.aux_reg()->next_id(),
            std::move(inp.scintillation),
            inp.auto_flush);
        actions.insert(scint_gen_action_);
    }

    // Create launch action with optical params+state and access to gen data
    detail::OpticalLaunchAction::Input la_inp;
    la_inp.model_builders = std::move(inp.model_builders);
    la_inp.material = inp.material;
    la_inp.offload = offload_params_;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.initializer_capacity = inp.initializer_capacity;
    launch_action_ = detail::OpticalLaunchAction::make_and_insert(
        core, std::move(la_inp));

    // Launch action must be *after* offload and generator actions
    CELER_ENSURE(!cherenkov_action_
                 || launch_action_->action_id()
                        > cherenkov_action_->action_id());
    CELER_ENSURE(!scint_action_
                 || launch_action_->action_id() > scint_action_->action_id());
    CELER_ENSURE(!cherenkov_gen_action_
                 || launch_action_->action_id()
                        > cherenkov_gen_action_->action_id());
    CELER_ENSURE(!scint_gen_action_
                 || launch_action_->action_id()
                        > scint_gen_action_->action_id());
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical generator data used for offloading.
 */
AuxId OpticalCollector::offload_aux_id() const
{
    return offload_params_->aux_id();
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical core state data.
 */
AuxId OpticalCollector::optical_aux_id() const
{
    return launch_action_->aux_id();
}

//---------------------------------------------------------------------------//
/*!
 * Get and reset cumulative statistics on optical generation from a state.
 */
OpticalAccumStats OpticalCollector::exchange_counters(AuxStateVec& aux) const
{
    auto& state = dynamic_cast<OpticalOffloadStateBase&>(
        aux.at(this->offload_aux_id()));
    return std::exchange(state.accum, {});
}

//---------------------------------------------------------------------------//
/*!
 * Get info on the number of tracks in the buffer.
 */
auto OpticalCollector::buffer_counts(AuxStateVec const& aux) const
    -> OpticalBufferSize const&
{
    auto& state = dynamic_cast<OpticalOffloadStateBase const&>(
        aux.at(this->offload_aux_id()));
    return state.buffer_size;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
