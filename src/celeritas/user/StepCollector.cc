//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.cc
//---------------------------------------------------------------------------//
#include "StepCollector.hh"

#include "celeritas/global/ActionRegistry.hh"

#include "detail/StepGatherAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and register pre and/or post-step actions.
 */
StepCollector::StepCollector(const StepSelection& selection,
                             SPStepInterface      callback,
                             ActionRegistry*      action_registry)
    : storage_(std::make_shared<detail::StepStorage>())
{
    CELER_EXPECT(action_registry);
    CELER_EXPECT(callback);

    {
        // Create params
        celeritas::HostVal<StepParamsData> host_data;
        host_data.selection = selection;
        storage_->params
            = CollectionMirror<StepParamsData>(std::move(host_data));
    }

    if (selection.pre_step)
    {
        // Pre-step data is being gathered (and no callback should be given)
        pre_action_
            = std::make_shared<detail::StepGatherAction<StepPoint::pre>>(
                action_registry->next_id(), storage_, nullptr);
        action_registry->insert(pre_action_);
    }

    // Always add post-step action, and add callback to it
    post_action_ = std::make_shared<detail::StepGatherAction<StepPoint::post>>(
        action_registry->next_id(), storage_, std::move(callback));
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
