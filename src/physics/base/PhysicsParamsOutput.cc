//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsParamsOutput.cc
//---------------------------------------------------------------------------//
#include "PhysicsParamsOutput.hh"

#include <utility>

#include "celeritas_config.h"
#include "base/Assert.hh"
#include "base/Range.hh"
#include "physics/base/Model.hh"
#include "sim/JsonPimpl.hh"

#include "PhysicsParams.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a shared action manager.
 */
PhysicsParamsOutput::PhysicsParamsOutput(SPConstPhysicsParams physics)
    : physics_(std::move(physics))
{
    CELER_EXPECT(physics_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void PhysicsParamsOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    // Save models
    {
        auto models = json::array();

        for (auto id : range(ModelId{physics_->num_models()}))
        {
            const Model& m = physics_->model(id);

            models.push_back(
                {{"label", m.label()},
                 {"process", physics_->process_id(id).unchecked_get()}});
        }
        obj["models"] = std::move(models);
    }

    // Save processes
    {
        auto processes = json::array();

        for (auto id : range(ProcessId{physics_->num_processes()}))
        {
            const Process& p = physics_->process(id);

            processes.push_back({{"label", p.label()}});
        }
        obj["processes"] = std::move(processes);
    }

    // Save options
    {
        const auto& scalars = physics_->host_ref().scalars;

        auto options = json::object();
#    define PPO_SAVE_OPTION(NAME) options[#    NAME] = scalars.NAME
        PPO_SAVE_OPTION(scaling_min_range);
        PPO_SAVE_OPTION(scaling_fraction);
        PPO_SAVE_OPTION(energy_fraction);
        PPO_SAVE_OPTION(linear_loss_limit);
        PPO_SAVE_OPTION(fixed_step_limiter);
        PPO_SAVE_OPTION(enable_fluctuation);
#    undef PPO_SAVE_OPTION
        obj["options"] = std::move(options);
    }

    // Save sizes
    {
        const auto& data = physics_->host_ref();

        auto sizes = json::object();
#    define PPO_SAVE_SIZE(NAME) sizes[#    NAME] = data.NAME.size()
        PPO_SAVE_SIZE(reals);
        PPO_SAVE_SIZE(model_ids);
        PPO_SAVE_SIZE(value_grids);
        PPO_SAVE_SIZE(value_grid_ids);
        PPO_SAVE_SIZE(process_ids);
        PPO_SAVE_SIZE(value_tables);
        PPO_SAVE_SIZE(integral_xs);
        PPO_SAVE_SIZE(model_groups);
        PPO_SAVE_SIZE(process_groups);
#    undef PPO_SAVE_SIZE
        obj["sizes"] = std::move(sizes);
    }

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
