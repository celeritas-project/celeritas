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
OpticalCollector::OpticalCollector(SPConstProperties properties,
                                   SPConstCerenkov cerenkov,
                                   SPConstScintillation scintillation,
                                   size_type buffer_capacity,
                                   size_type num_streams,
                                   ActionRegistry* action_registry)
    : storage_(std::make_shared<detail::OpticalGenStorage>())
{
    CELER_EXPECT(scintillation || (cerenkov && properties));
    CELER_EXPECT(buffer_capacity > 0);
    CELER_EXPECT(num_streams > 0);
    CELER_EXPECT(action_registry);

    // Create params and stream storage
    HostVal<OpticalGenParamsData> host_data;
    host_data.cerenkov = cerenkov && properties;
    host_data.scintillation = static_cast<bool>(scintillation);
    host_data.capacity = buffer_capacity;
    storage_->obj = {std::move(host_data), num_streams};
    storage_->size.resize(num_streams, {});

    // Action to gather pre-step data needed to generate optical distributions
    gather_action_ = std::make_shared<detail::PreGenGatherAction>(
        action_registry->next_id(), storage_);
    action_registry->insert(gather_action_);

    // Action to generate Cerenkov and scintillation optical distributions
    pregen_action_
        = std::make_shared<detail::PreGenAction>(action_registry->next_id(),
                                                 properties,
                                                 cerenkov,
                                                 scintillation,
                                                 storage_);
    action_registry->insert(pregen_action_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
