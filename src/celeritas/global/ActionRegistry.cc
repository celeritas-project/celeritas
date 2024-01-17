//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistry.cc
//---------------------------------------------------------------------------//
#include "ActionRegistry.hh"

#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Add action to mutable list if it has any mutable member functions.
 */
void ActionRegistry::insert_mutable_impl(SPAction&& action)
{
    CELER_EXPECT(action);
    if (dynamic_cast<BeginRunActionInterface*>(action.get()))
    {
        mutable_actions_.push_back(action);
    }
    return this->insert_impl(std::move(action));
}

//---------------------------------------------------------------------------//
/*!
 * Perform checks on an immutable action before inserting.
 */
void ActionRegistry::insert_const_impl(SPConstAction&& action)
{
    CELER_EXPECT(action);
    CELER_VALIDATE(!dynamic_cast<BeginRunActionInterface const*>(action.get()),
                   << "begin-run action '" << action->label()
                   << "' (ID=" << action->action_id().unchecked_get()
                   << ") cannot be registered as const");
    return this->insert_impl(std::move(action));
}

//---------------------------------------------------------------------------//
/*!
 * Register an action.
 */
void ActionRegistry::insert_impl(SPConstAction&& action)
{
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
ActionId ActionRegistry::find_action(std::string const& label) const
{
    auto iter = action_ids_.find(label);
    if (iter == action_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
