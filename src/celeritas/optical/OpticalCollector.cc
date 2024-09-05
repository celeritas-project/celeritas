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
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CerenkovParams.hh"
#include "CoreParams.hh"
#include "MaterialParams.hh"
#include "OffloadData.hh"
#include "ScintillationParams.hh"

#include "detail/CerenkovOffloadAction.hh"
#include "detail/OffloadGatherAction.hh"
#include "detail/OffloadParams.hh"
#include "detail/OpticalLaunchAction.hh"
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
    gen_params_ = std::make_shared<detail::OffloadParams>(aux.next_id(), setup);
    aux.insert(gen_params_);

    // Action to gather pre-step data needed to generate optical distributions
    ActionRegistry& actions = *core.action_reg();
    gather_action_ = std::make_shared<detail::OffloadGatherAction>(
        actions.next_id(), gen_params_->aux_id());
    actions.insert(gather_action_);

    if (setup.cerenkov)
    {
        // Action to generate Cerenkov optical distributions
        cerenkov_action_ = std::make_shared<detail::CerenkovOffloadAction>(
            actions.next_id(),
            gen_params_->aux_id(),
            inp.material,
            std::move(inp.cerenkov));
        actions.insert(cerenkov_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_action_ = std::make_shared<detail::ScintOffloadAction>(
            actions.next_id(),
            gen_params_->aux_id(),
            std::move(inp.scintillation));
        actions.insert(scint_action_);
    }

    // Create launch action with optical params+state and access to gen data
    launch_action_ = detail::OpticalLaunchAction::make_and_insert(
        core, inp.material, gen_params_);

    // Launch action must be *after* generator actions
    CELER_ENSURE(!cerenkov_action_
                 || launch_action_->action_id()
                        > cerenkov_action_->action_id());
    CELER_ENSURE(!scint_action_
                 || launch_action_->action_id() > scint_action_->action_id());
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical generator data used for offloading.
 */
AuxId OpticalCollector::offload_aux_id() const
{
    return gen_params_->aux_id();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
