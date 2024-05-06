//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalGenData.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "detail/OpticalGenStorage.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical params, number of streams, and action registry.
 */
OpticalCollector::OpticalCollector(Input inp)
    : storage_(std::make_shared<detail::OpticalGenStorage>())
{
    CELER_EXPECT(inp);

    // Create params and stream storage
    HostVal<OpticalGenParamsData> host_data;
    host_data.cerenkov = inp.cerenkov && inp.properties;
    host_data.scintillation = static_cast<bool>(inp.scintillation);
    host_data.capacity = inp.buffer_capacity;
    storage_->obj = {std::move(host_data), inp.num_streams};
    storage_->size.resize(inp.num_streams, {});

    // Action to gather pre-step data needed to generate optical distributions
    gather_action_ = std::make_shared<detail::PreGenGatherAction>(
        inp.action_registry->next_id(), storage_);
    inp.action_registry->insert(gather_action_);

    // Action to generate Cerenkov and scintillation optical distributions
    pregen_action_ = std::make_shared<detail::PreGenAction>(
        inp.action_registry->next_id(),
        inp.properties,
        inp.cerenkov,
        inp.scintillation,
        storage_);
    inp.action_registry->insert(pregen_action_);

    // TODO: add an action to launch optical tracking loop
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
