//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SharedActionSetBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
template<class Label>
class SharedActionSetBuilder
{
  public:
    // Construct with defaults
    inline SharedActionSetBuilder(ActionRegistry*);

    // See if action is registered under this label
    inline ActionId operator()(Label label) const;

    // See if action is registered under this label, and if not build and
    // register it
    template<class F>
    inline ActionId operator()(Label label, F&& builder);

  private:
    ActionRegistry* action_reg_;
    std::map<Label, ActionId> actions_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
template<class Label>
SharedActionSetBuilder::SharedActionSetBuilder(ActionRegistry* reg)
    : action_reg_(reg)
{
    CELER_EXPECT(action_reg_);
}

template<class Label>
ActionId SharedActionSetBuilder::operator()(Label label) const
{
    auto iter = actions_.find(label);
    if (iter != actions_.end())
    {
        return iter->second;
    }
    else
    {
        return ActionId{};
    }
}

template<class Label>
template<class F>
ActionId SharedActionSetBuilder::operator()(Label label, F&& builder)
{
    auto iter = actions_.find(label);
    if (iter != actions_.end())
    {
        return iter->second;
    }
    else
    {
        auto action_id = action_reg_->next_id();
        auto action = builder(action_id);

        CELER_ENSURE(action);
        CELER_ENSURE(action->action_id() == action_id);

        action_reg_->insert(action);
        return action_id;
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
