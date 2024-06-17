//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalGenData.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "detail/OpticalGenStorage.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical params, number of streams, and core data.
 */
OpticalCollector::OpticalCollector(CoreParams const& core, Input&& inp)
    : storage_(std::make_shared<detail::OpticalGenStorage>())
{
    CELER_EXPECT(inp);

    size_type num_streams = core.max_streams();
    ActionRegistry& actions = *core.action_reg();

    OpticalGenSetup setup;
    setup.cerenkov = inp.cerenkov && inp.properties;
    setup.scintillation = static_cast<bool>(inp.scintillation);
    setup.capacity = inp.buffer_capacity;

    // Create params and stream storage
    storage_->obj = {HostVal<OpticalGenParamsData>{setup}, num_streams};
    storage_->size.resize(num_streams, {});

    // Action to gather pre-step data needed to generate optical distributions
    gather_action_ = std::make_shared<detail::PreGenGatherAction>(
        actions.next_id(), storage_);
    actions.insert(gather_action_);

    if (setup.cerenkov)
    {
        // Action to generate Cerenkov optical distributions
        cerenkov_pregen_action_
            = std::make_shared<detail::CerenkovPreGenAction>(
                actions.next_id(),
                std::move(inp.properties),
                std::move(inp.cerenkov),
                storage_);
        actions.insert(cerenkov_pregen_action_);
    }

    if (setup.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_pregen_action_ = std::make_shared<detail::ScintPreGenAction>(
            actions.next_id(), std::move(inp.scintillation), storage_);
        actions.insert(scint_pregen_action_);
    }

    // TODO: add an action to launch optical tracking loop
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
