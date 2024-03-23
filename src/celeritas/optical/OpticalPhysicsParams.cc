//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysicsParams.cc
//---------------------------------------------------------------------------//
#include "OpticalPhysicsParams.hh"

#include "corecel/sys/ScopedMem.hh"
#include "celeritas/global/ActionRegistry.hh"

#include "OpticalModel.hh"
#include "OpticalProcess.hh"

namespace celeritas
{
class ImplicitPhysicsAction final : public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};

//---------------------------------------------------------------------------//
OpticalPhysicsParams::OpticalPhysicsParams(Input input)
    : processes_(std::move(input.processes))
{
    CELER_EXPECT(!processes_.empty());
    CELER_EXPECT(std::all_of(processes_.begin(),
                             processes_.end(),
                             [](SPConstProcess const& p) { return bool(p); }));
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.action_registry);

    ScopedMem record_mem("OpticalPhysicsParams.construct");

    // Create actions
    {
        using std::make_shared;
        auto& action_reg = *input.action_registry;

        // TODO: Add more actions?

        discrete_action_ = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "discrete-action",
            "placeholder for discrete optical action");
        action_reg.insert(discrete_action_);

        models_ = build_models(action_reg);

        failure_action_ = std::make_shared<ImplicitPhysicsAction>(
                action_reg.next_id(),
                "physics-failure",
                "mark a track that failed to sample an interaction");
        action_reg.insert(failure_action_);
    }

    HostValue host_data;
    this->build_options(input.options, &host_data);
    this->build_lambda(*input.materials, &host_data);

    data_ = CollectionMirror<OpticalPhysicsParamsData>{std::move(host_data)};

    CELER_ENSURE(discrete_action_->action_id() == host_ref().scalars.discrete_action());
}

auto OpticalPhysicsParams::build_models(ActionRegistry& reg) const -> VecModel
{
    VecModel models;

    for (auto process_idx : range<OpticalProcessId::size_type>(processes_.size()))
    {
        auto id_iter = OpticalProcess::ActionIdIter{reg.next_id()};
        auto model = processes_[process_idx]->build_model(id_iter);
        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == *id_iter++);

        reg.insert(model);
        models.push_back({std::move(model), OpticalProcessId{process_idx}});
    }

    CELER_ENSURE(!models.empty());
    return models;
}

void OpticalPhysicsParams::build_options(Options const& opts, HostValue* data) const
{
}

void OpticalPhysicsParams::build_lambda(MaterialParams const& mats, HostValue* data) const
{
    CELER_EXPECT(*data);

    auto value_tables = make_builder(&data->value_tables);
    auto value_grid_ids = make_builder(&data->value_grid_ids);

    GenericGridInserter insert_grid(&data->reals, &data->value_grids);
    
    // Material dependent physics tables - one per optical process
    std::vector<OpticalValueTable> temp_tables;
    temp_tables.resize(num_processes());

    // Loop over each optical process
    for (auto process_idx : range(OpticalProcessId::size_type(processes_.size())))
    {
        OpticalProcess const& proc = *this->processes_[process_idx];

        // Grid IDs of lambda grid for each material
        std::vector<OpticalValueGridId> temp_grid_ids
            = proc.step_limits(insert_grid);

        if (std::any_of(temp_grid_ids.begin(),
                         temp_grid_ids.end(),
                         [](OpticalValueGridId id) { return bool(id); }))
        {
            // Construct optical value grid table
            OpticalValueTable& temp_table = temp_tables[process_idx];
            temp_table.grids = value_grid_ids.insert_back(temp_grid_ids.begin(), temp_grid_ids.end());
            CELER_ASSERT(temp_table.grids.size() == mats.size());
        }
    }

    // Construct value tables
    data->process_groups.macro_xs_tables
        = value_tables.insert_back(temp_tables.begin(), temp_tables.end());
    CELER_ASSERT(data->process_groups.macro_xs_tables.size()
                 == processes_.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
