//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistryOutput.cc
//---------------------------------------------------------------------------//
#include "ActionRegistryOutput.hh"

#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"

#include "ActionInterface.hh"
#include "ActionRegistry.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared action manager.
 */
ActionRegistryOutput::ActionRegistryOutput(SPConstActionRegistry actions)
    : actions_(std::move(actions))
{
    CELER_EXPECT(actions_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ActionRegistryOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto label = json::array();
    auto description = json::array();

    for (auto id : range(ActionId{actions_->num_actions()}))
    {
        label.push_back(actions_->id_to_label(id));

        ActionInterface const& action = *actions_->action(id);
        description.push_back(action.description());
    }
    j->obj = {
        {"label", std::move(label)},
        {"description", std::move(description)},
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
