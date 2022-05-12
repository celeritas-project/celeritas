//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct CoreRef;

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
 * next_id . Registering an action checks its ID.
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

    //! Construction/execution options
    struct Options
    {
        bool sync{false}; //!< Call DeviceSynchronize and add timer
    };

  public:
    //! Construct with options
    explicit ActionManager(Options options) : options_(options) {}

    //! Construct with default options
    ActionManager() : ActionManager(Options{}) {}

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

    //! Whether synchronization is taking place
    bool sync() const { return options_.sync; }

    // Get the number of defined actions
    inline ActionId::size_type num_actions() const;

    // Access an action
    inline const ActionInterface& action(ActionId id) const;

    // Get the label corresponding to an action
    inline const std::string& id_to_label(ActionId id) const;

    // Get the accumulated launch time if syncing is enabled
    inline double accum_time(ActionId id) const;

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
        mutable double time{0};
    };

    //// DATA ////

    Options                                   options_;
    std::vector<ActionData>                   actions_;
    std::unordered_map<std::string, ActionId> action_ids_;

    //// HELPER_FUNCTIONS ////
    void insert_impl(SPConstAction&& action, PConstExplicit expl);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
/*!
 * Get the accumulated launch time if syncing is enabled.
 */
double ActionManager::accum_time(ActionId id) const
{
    CELER_EXPECT(this->sync());
    CELER_EXPECT(id < actions_.size());
    return actions_[id.unchecked_get()].time;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
