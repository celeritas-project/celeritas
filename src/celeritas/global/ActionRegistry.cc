//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistry.cc
//---------------------------------------------------------------------------//
#include "ActionRegistry.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Stopwatch.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Register an implicit action.
 */
void ActionRegistry::insert(SPConstImplicit action)
{
    CELER_EXPECT(action);
    this->insert_impl(std::move(action), nullptr);
}

//---------------------------------------------------------------------------//
/*!
 * Register an explicit action.
 */
void ActionRegistry::insert(SPConstExplicit action)
{
    CELER_EXPECT(action);
    // Get explicit action pointer
    PConstExplicit expl = action.get();
    this->insert_impl(std::move(action), expl);
}

//---------------------------------------------------------------------------//
/*!
 * Call the given action ID with host or device data.
 *
 * The given action ID \em must be an explicit action.
 */
template<MemSpace M>
void ActionRegistry::invoke(ActionId id, const CoreRef<M>& data) const
{
    CELER_ASSERT(id < actions_.size());
    const auto& action_data = actions_[id.unchecked_get()];
    CELER_VALIDATE(action_data.expl,
                   << "action '" << actions_[id.unchecked_get()].label
                   << "' is implicit and cannot be invoked");

    if (options_.sync)
    {
        Stopwatch get_time;
        action_data.expl->execute(data);
        if (M == MemSpace::device)
        {
            CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
        }
        action_data.time += get_time();
    }
    else
    {
        action_data.expl->execute(data);
    }
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
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
void ActionRegistry::insert_impl(SPConstAction&& action, PConstExplicit expl)
{
    CELER_ASSERT(action);

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

    actions_.emplace_back(
        ActionData{std::move(label), std::move(action), expl});
}

//---------------------------------------------------------------------------//
// Explicit template instantiation
//---------------------------------------------------------------------------//

template void
ActionRegistry::invoke(ActionId, const CoreRef<MemSpace::host>&) const;
template void
ActionRegistry::invoke(ActionId, const CoreRef<MemSpace::device>&) const;

//---------------------------------------------------------------------------//
} // namespace celeritas
