//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CerenkovParams.hh"
#include "CoreParams.hh"
#include "MaterialParams.hh"
#include "OffloadData.hh"
#include "ScintillationParams.hh"

#include "detail/CerenkovGeneratorAction.hh"
#include "detail/CerenkovOffloadAction.hh"
#include "detail/OffloadGatherAction.hh"
#include "detail/OffloadParams.hh"
#include "detail/OpticalLaunchAction.hh"
#include "detail/ScintGeneratorAction.hh"
#include "detail/ScintOffloadAction.hh"

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
    setup.cerenkov = inp.cerenkov && inp.material;
    setup.scintillation = static_cast<bool>(inp.scintillation);
    setup.capacity = inp.buffer_capacity;

    // Create offload params
    AuxParamsRegistry& aux = *core.aux_reg();
    offload_params_
        = std::make_shared<detail::OffloadParams>(aux.next_id(), setup);
    aux.insert(offload_params_);

    // Action to gather pre-step data needed to generate optical distributions
    ActionRegistry& actions = *core.action_reg();
    gather_action_ = std::make_shared<detail::OffloadGatherAction>(
        actions.next_id(), offload_params_->aux_id());
    actions.insert(gather_action_);

    if (setup.cerenkov)
    {
        // Action to generate Cerenkov optical distributions
        cerenkov_action_ = std::make_shared<detail::CerenkovOffloadAction>(
            actions.next_id(),
            offload_params_->aux_id(),
            inp.material,
            inp.cerenkov);
        actions.insert(cerenkov_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_action_ = std::make_shared<detail::ScintOffloadAction>(
            actions.next_id(), offload_params_->aux_id(), inp.scintillation);
        actions.insert(scint_action_);
    }

    if (setup.cerenkov)
    {
        // Action to generate Cerenkov primaries
        cerenkov_gen_action_
            = std::make_shared<detail::CerenkovGeneratorAction>(
                actions.next_id(),
                offload_params_->aux_id(),
                // TODO: Hack: generator action must be before launch action
                // but needs optical state aux ID
                core.aux_reg()->next_id(),
                inp.material,
                std::move(inp.cerenkov),
                inp.auto_flush);
        actions.insert(cerenkov_gen_action_);
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
    la_inp.material = inp.material;
    la_inp.offload = offload_params_;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.initializer_capacity = inp.initializer_capacity;
    launch_action_ = detail::OpticalLaunchAction::make_and_insert(
        core, std::move(la_inp));

    // Launch action must be *after* offload and generator actions
    CELER_ENSURE(!cerenkov_action_
                 || launch_action_->action_id()
                        > cerenkov_action_->action_id());
    CELER_ENSURE(!scint_action_
                 || launch_action_->action_id() > scint_action_->action_id());
    CELER_ENSURE(!cerenkov_gen_action_
                 || launch_action_->action_id()
                        > cerenkov_gen_action_->action_id());
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
}  // namespace celeritas
