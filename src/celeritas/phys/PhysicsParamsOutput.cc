//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsParamsOutput.cc
//---------------------------------------------------------------------------//
#include "PhysicsParamsOutput.hh"

#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/math/QuantityIO.json.hh"

#include "Model.hh"
#include "PhysicsData.hh"
#include "PhysicsParams.hh"  // IWYU pragma: keep
#include "Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared physics data.
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
    using json = nlohmann::json;

    auto obj = json::object();

    // Save models
    {
        auto label = json::array();
        auto process_id = json::array();

        for (auto id : range(ModelId{physics_->num_models()}))
        {
            Model const& m = *physics_->model(id);
            label.push_back(m.label());
            process_id.push_back(physics_->process_id(id).unchecked_get());
        }
        obj["models"] = {
            {"label", std::move(label)},
            {"process_id", std::move(process_id)},
        };
    }

    // Save processes
    {
        auto label = json::array();

        for (auto id : range(ProcessId{physics_->num_processes()}))
        {
            Process const& p = *physics_->process(id);
            label.push_back(p.label());
        }
        obj["processes"] = {{"label", std::move(label)}};
    }

    // Save options
    {
        auto const& scalars = physics_->host_ref().scalars;

        auto options = json::object();
#define PPO_SAVE_OPTION(NAME) options[#NAME] = scalars.NAME
        PPO_SAVE_OPTION(light.min_range);
        PPO_SAVE_OPTION(heavy.min_range);
        PPO_SAVE_OPTION(light.max_step_over_range);
        PPO_SAVE_OPTION(heavy.max_step_over_range);
        PPO_SAVE_OPTION(min_eprime_over_e);
        PPO_SAVE_OPTION(light.lowest_energy);
        PPO_SAVE_OPTION(heavy.lowest_energy);
        PPO_SAVE_OPTION(linear_loss_limit);
        PPO_SAVE_OPTION(fixed_step_limiter);
#undef PPO_SAVE_OPTION
        obj["options"] = std::move(options);
    }

    // Save sizes
    {
        auto const& data = physics_->host_ref();

        auto sizes = json::object();
#define PPO_SAVE_SIZE(NAME) sizes[#NAME] = data.NAME.size()
        PPO_SAVE_SIZE(reals);
        PPO_SAVE_SIZE(model_ids);
        PPO_SAVE_SIZE(xs_grids);
        PPO_SAVE_SIZE(xs_grid_ids);
        PPO_SAVE_SIZE(xs_tables);
        PPO_SAVE_SIZE(uniform_grids);
        PPO_SAVE_SIZE(uniform_grid_ids);
        PPO_SAVE_SIZE(uniform_tables);
        PPO_SAVE_SIZE(process_ids);
        PPO_SAVE_SIZE(integral_xs);
        PPO_SAVE_SIZE(model_groups);
        PPO_SAVE_SIZE(process_groups);
#undef PPO_SAVE_SIZE
        obj["sizes"] = std::move(sizes);
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
