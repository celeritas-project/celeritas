//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/MaterialPropertyParams.hh"
#include "celeritas/optical/OffloadData.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "detail/OffloadParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with core data and optical data.
 */
OpticalCollector::OpticalCollector(CoreParams const& core, Input&& inp)
{
    CELER_EXPECT(inp);

    OffloadOptions setup;
    setup.cerenkov = inp.cerenkov && inp.properties;
    setup.scintillation = static_cast<bool>(inp.scintillation);
    setup.capacity = inp.buffer_capacity;

    // Create aux params and add to core
    gen_params_ = std::make_shared<detail::OffloadParams>(
        core.aux_reg()->next_id(), setup);
    core.aux_reg()->insert(gen_params_);

    // Action to gather pre-step data needed to generate optical distributions
    ActionRegistry& actions = *core.action_reg();
    gather_action_ = std::make_shared<detail::OffloadGatherAction>(
        actions.next_id(), gen_params_->aux_id());
    actions.insert(gather_action_);

    if (setup.cerenkov)
    {
        // Action to generate Cerenkov optical distributions
        cerenkov_offload_action_
            = std::make_shared<detail::CerenkovOffloadAction>(
                actions.next_id(),
                gen_params_->aux_id(),
                std::move(inp.properties),
                std::move(inp.cerenkov));
        actions.insert(cerenkov_offload_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_offload_action_ = std::make_shared<detail::ScintOffloadAction>(
            actions.next_id(),
            gen_params_->aux_id(),
            std::move(inp.scintillation));
        actions.insert(scint_offload_action_);
    }

    // TODO: add an action to launch optical tracking loop
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical generator data.
 */
AuxId OpticalCollector::aux_id() const
{
    return gen_params_->aux_id();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
