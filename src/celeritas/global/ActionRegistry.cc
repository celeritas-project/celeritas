//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistry.cc
//---------------------------------------------------------------------------//
#include "ActionRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Register an implicit action.
 */
void ActionRegistry::insert(SPConstAction action)
{
    CELER_EXPECT(action);

    auto label = action->label();
    CELER_VALIDATE(!label.empty(), << "action label is empty");

    auto id = action->action_id();
    CELER_VALIDATE(id == this->next_id(),
                   << "incorrect action id {" << id.unchecked_get()
                   << "} for action '" << label << "' (should be {"
                   << this->next_id().get() << "})");

    auto iter_inserted = action_ids_.insert({label, id});
    CELER_VALIDATE(iter_inserted.second,
                   << "duplicate action label '" << label << "'");

    actions_.push_back(std::move(action));
    labels_.push_back(std::move(label));

    CELER_ENSURE(action_ids_.size() == actions_.size());
    CELER_ENSURE(labels_.size() == actions_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Find the action corresponding to an label.
 */
ActionId ActionRegistry::find_action(const std::string& label) const
{
    auto iter = action_ids_.find(label);
    if (iter == action_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
