//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.cc
//---------------------------------------------------------------------------//
#include "StepCollector.hh"

#include <algorithm>

#include "celeritas/global/ActionRegistry.hh"

#include "detail/StepGatherAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and register pre and/or post-step actions.
 */
StepCollector::StepCollector(VecInterface    callbacks,
                             ActionRegistry* action_registry)
    : storage_(std::make_shared<detail::StepStorage>())
{
    CELER_EXPECT(action_registry);
    CELER_EXPECT(!callbacks.empty());
    CELER_EXPECT(std::all_of(
        callbacks.begin(), callbacks.end(), [](const SPStepInterface& i) {
            return static_cast<bool>(i);
        }));

    // Loop over callbacks to take union of step selections
    StepSelection selection;
    {
        CELER_ASSERT(!selection);
        for (const SPStepInterface& sp_interface : callbacks)
        {
            auto this_selection = sp_interface->selection();
            CELER_VALIDATE(this_selection,
                           << "step interface doesn't collect any data");
            selection |= this_selection;
        }
        CELER_ASSERT(selection);
    }

    {
        // Create params
        celeritas::HostVal<StepParamsData> host_data;

        host_data.selection = selection;
        storage_->params
            = CollectionMirror<StepParamsData>(std::move(host_data));
    }

    if (selection.points[StepPoint::pre])
    {
        // Some pre-step data is being gathered
        pre_action_
            = std::make_shared<detail::StepGatherAction<StepPoint::pre>>(
                action_registry->next_id(), storage_, VecInterface{});
        action_registry->insert(pre_action_);
    }

    // Always add post-step action, and add callbacks to it
    post_action_ = std::make_shared<detail::StepGatherAction<StepPoint::post>>(
        action_registry->next_id(), storage_, std::move(callbacks));
    action_registry->insert(post_action_);
}

//---------------------------------------------------------------------------//
//!@{
//! Default destructor and move
StepCollector::~StepCollector()                          = default;
StepCollector::StepCollector(StepCollector&&)            = default;
StepCollector& StepCollector::operator=(StepCollector&&) = default;
//!@}

//---------------------------------------------------------------------------//
/*!
 * See which data are being gathered.
 */
const StepSelection& StepCollector::selection() const
{
    return storage_->params.host_ref().selection;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
