//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and store metadata about end-of-step actions.
 *
 * The action manager helps create and retain access to "actions" (possible
 * control paths for a track) over the program's lifetime. "Implicit" actions
 * are primarily for debugging, but "explicit" actions indicate that a kernel
 * will change the state of a track on device.
 *
 * Associated actions use the \c ActionInterface class to provide debugging
 * information, and the \c ExplicitActionInterface is used to invoke kernels
 * with core data.
 *
 * New actions should be created with an action ID corresponding to \c
 * next_id . Registering an action checks its ID. Actions are always added
 * sequentially and can never be removed from the registry once added.
 */
class ActionRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using SPAction = std::shared_ptr<ActionInterface>;
    using SPConstAction = std::shared_ptr<ActionInterface const>;
    //!@}

  public:
    //! Construct without any registered actions
    ActionRegistry() = default;

    //// CONSTRUCTION ////

    //! Get the next available action ID
    ActionId next_id() const { return ActionId(actions_.size()); }

    // Register a mutable action
    void insert_mutable(SPAction);

    // Register a const action
    void insert(SPConstAction);

    //// ACCESSORS ////

    //! Get the number of defined actions
    ActionId::size_type num_actions() const { return actions_.size(); }

    //! Whether no actions have been registered
    bool empty() const { return actions_.empty(); }

    // Access an action
    inline SPConstAction const& action(ActionId id) const;

    // Get the label corresponding to an action
    inline std::string const& id_to_label(ActionId id) const;

    // Find the action corresponding to an label
    ActionId find_action(std::string const& label) const;

    // View all actions that are able to change at runtime
    inline Span<SPAction const> mutable_actions() const;

  private:
    //// DATA ////

    std::vector<SPConstAction> actions_;
    std::vector<std::string> labels_;
    std::unordered_map<std::string, ActionId> action_ids_;
    std::vector<SPAction> mutable_actions_;

    void insert_impl(SPConstAction&&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access an action.
 */
auto ActionRegistry::action(ActionId id) const -> SPConstAction const&
{
    CELER_EXPECT(id < actions_.size());
    return actions_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label corresponding to an action.
 */
std::string const& ActionRegistry::id_to_label(ActionId id) const
{
    CELER_EXPECT(id < actions_.size());
    return labels_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * View all actions that are able to change at runtime.
 */
auto ActionRegistry::mutable_actions() const -> Span<SPAction const>
{
    return make_span(mutable_actions_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
