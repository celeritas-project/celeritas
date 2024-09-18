//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionGroups.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/cont/Range.hh"

#include "ActionRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from an action registry.
 */
template<class P, template<MemSpace M> class S>
ActionGroups<P, S>::ActionGroups(ActionRegistry const& reg)
{
    // Loop over all action IDs
    for (auto aidx : range(reg.num_actions()))
    {
        // Get abstract action shared pointer to determine type
        auto const& base = reg.action(ActionId{aidx});
        if (auto step_act = std::dynamic_pointer_cast<StepActionT const>(base))
        {
            // Add stepping action to our array
            step_actions_.push_back(std::move(step_act));
        }
    }

    // Loop over all mutable actions
    for (auto const& base : reg.mutable_actions())
    {
        if (auto brun = std::dynamic_pointer_cast<BeginRunActionT>(base))
        {
            // Add beginning-of-run to the array
            begin_run_.emplace_back(std::move(brun));
        }
    }

    // Sort actions by increasing order (and secondarily, increasing IDs)
    std::sort(step_actions_.begin(),
              step_actions_.end(),
              [](SPConstStepAction const& a, SPConstStepAction const& b) {
                  return OrderedAction{a->order(), a->action_id()}
                         < OrderedAction{b->order(), b->action_id()};
              });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
