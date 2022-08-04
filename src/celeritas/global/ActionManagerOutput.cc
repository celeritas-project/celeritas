//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionManagerOutput.cc
//---------------------------------------------------------------------------//
#include "ActionManagerOutput.hh"

#include <utility>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "ActionManager.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared action manager.
 */
ActionManagerOutput::ActionManagerOutput(SPConstActionManager actions)
    : actions_(std::move(actions))
{
    CELER_EXPECT(actions_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ActionManagerOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    auto obj        = nlohmann::json::array();
    bool has_timing = actions_->sync();
    for (auto id : range(ActionId{actions_->num_actions()}))
    {
        nlohmann::json entry{
            {"label", actions_->id_to_label(id)},
        };

        const ActionInterface& action = actions_->action(id);
        std::string            desc   = action.description();
        if (!desc.empty())
        {
            entry["description"] = std::move(desc);
        }
        if (has_timing)
        {
            // TODO: only output if explicit
            entry["time"] = actions_->accum_time(id);
        }
        obj.push_back(entry);
    }
    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
