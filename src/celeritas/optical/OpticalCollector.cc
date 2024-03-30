//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalGenData.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "detail/GenStorage.hh"
#include "detail/PreGenAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical params, number of streams, and action registry.
 */
OpticalCollector::OpticalCollector(SPConstProperties properties,
                                   SPConstCerenkov cerenkov,
                                   SPConstScintillation scintillation,
                                   size_type num_streams,
                                   ActionRegistry* action_registry)
    : storage_(std::make_shared<detail::GenStorage>())
{
    CELER_EXPECT(scintillation || (cerenkov && properties));
    CELER_EXPECT(num_streams > 0);
    CELER_EXPECT(action_registry);

    // Create params
    HostVal<OpticalGenParamsData> host_data;
    if (cerenkov)
    {
        host_data.cerenkov = get_ref<MemSpace::native>(*cerenkov);
        host_data.properties = get_ref<MemSpace::native>(*properties);
    }
    if (scintillation)
    {
        host_data.scintillation = get_ref<MemSpace::native>(*scintillation);
    }
    storage_->obj = {std::move(host_data), num_streams};

    // Build action to gather pre-step data needed for generating optical
    // distributions
    gather_action_ = std::make_shared<detail::PreGenAction<StepPoint::pre>>(
        action_registry->next_id(), storage_);
    action_registry->insert(gather_action_);

    // Build action to generate optical distribution data from pre-step and
    // state data
    pregen_action_ = std::make_shared<detail::PreGenAction<StepPoint::post>>(
        action_registry->next_id(), storage_);
    action_registry->insert(pregen_action_);
}

//---------------------------------------------------------------------------//
//!@{
//! Default destructor and move and copy
OpticalCollector::~OpticalCollector() = default;
OpticalCollector::OpticalCollector(OpticalCollector const&) = default;
OpticalCollector& OpticalCollector::operator=(OpticalCollector const&)
    = default;
OpticalCollector::OpticalCollector(OpticalCollector&&) = default;
OpticalCollector& OpticalCollector::operator=(OpticalCollector&&) = default;
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
