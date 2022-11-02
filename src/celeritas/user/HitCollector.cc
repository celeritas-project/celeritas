//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HitCollector.cc
//---------------------------------------------------------------------------//
#include "HitCollector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and register pre and/or post-step actions.
 */
HitCollector::HitCollector(const HitSelection& selection,
                           Callbacks           cb,
                           ActionRegistry*     action_registry)
    : selection_(selection), buffer_(std::make_shared<detail::HitBuffer>())
{
    CELER_EXPECT(action_registry);
    CELER_EXPECT(cb.pre_step || cb.post_step);

    // Construct shared "params" data
    if (!cb.post_step)
    {
        selection_.post_step = false;
    }

    // Create actions
    if (cb.pre_step)
    {
        pre_action_ = std::make_shared<HitAction>(action_registry->next_id(),
                                                  selection_,
                                                  buffer_,
                                                  std::move(cb.pre_step),
                                                  Action::pre);
        action_registry->insert(pre_action_);
    }
    if (cb.post_step)
    {
        post_action_ = std::make_shared<HitAction>(action_registry->next_id(),
                                                   selection_,
                                                   buffer_,
                                                   std::move(cb.pre_step), );
        action_registry->insert(post_action_);
    }
}

//---------------------------------------------------------------------------//
//! Default destructor
HitCollector::~HitCollector() = default;

//---------------------------------------------------------------------------//
} // namespace celeritas
