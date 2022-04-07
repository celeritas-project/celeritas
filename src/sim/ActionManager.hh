//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ActionManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/Range.hh"

#include "ActionInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and store metadata about end-of-step actions.
 *
 * Registering an action checks its ID.
 */
class ActionManager
{
  public:
    //!@{
    //! Type aliases
    using ActionRange     = Range<ActionId>;
    using SPConstImplicit = std::shared_ptr<const ImplicitActionInterface>;
    using SPConstExplicit = std::shared_ptr<const ExplicitActionInterface>;
    //!@}

  public:
    //// CONSTRUCTION ////

    // Get the next action ID
    inline ActionId next_id() const;

    // Register an implicit action
    void insert(SPConstImplicit);

    // Register an explicit action
    void insert(SPConstExplicit);

    //// INVOCATION ////

    // Call the given action ID with host or device data.
    template<MemSpace M>
    void invoke(ActionId explicit_id, const CoreRef<M>& data) const;

    //// ACCESSORS ////

    // Get the number of defined actions
    inline ActionId::size_type num_actions() const;

    // Access an action
    inline const ActionInterface& action(ActionId id) const;

    // Get the label corresponding to an action
    inline const std::string& id_to_label(ActionId id) const;

    // Find the action corresponding to an label
    ActionId find_action(const std::string& label) const;

  private:
    //// TYPES ////

    using SPConstAction  = std::shared_ptr<const ActionInterface>;
    using PConstExplicit = ExplicitActionInterface const*;

    struct ActionData
    {
        std::string    label;
        SPConstAction  action;
        PConstExplicit expl{nullptr}; //!< dynamic_cast of action
    };

    //// DATA ////

    std::vector<ActionData>                   actions_;
    std::vector<ActionRange>                  ranges_;
    std::unordered_map<std::string, ActionId> action_ids_;

    //// HELPER_FUNCTIONS ////
    void insert_impl(SPConstAction&& action, PConstExplicit expl);
};

//---------------------------------------------------------------------------//
/*!
 * Get the next available action ID.
 */
ActionId ActionManager::next_id() const
{
    return ActionId(actions_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of defined actions.
 */
ActionId::size_type ActionManager::num_actions() const
{
    return actions_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Access an action.
 */
const ActionInterface& ActionManager::action(ActionId id) const
{
    CELER_EXPECT(id < actions_.size());
    return *actions_[id.unchecked_get()].action;
}

//---------------------------------------------------------------------------//
/*!
 * Get the label corresponding to an action.
 */
const std::string& ActionManager::id_to_label(ActionId id) const
{
    CELER_EXPECT(id < actions_.size());
    return actions_[id.unchecked_get()].label;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
