//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

#include "corecel/sys/ActionRegistry.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
SurfacePhysicsParams::SurfacePhysicsParams(Input input)
{
    CELER_EXPECT(!input.model_builders.empty());
    CELER_EXPECT(input.action_registry);

    // Create and register actions
    {
        auto& action_reg = *input.action_registry;

        // TODO: build any explicit actions?

        // Build models
        models_ = this->build_models(input.model_builders, action_reg);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;
    
    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 */
auto SurfacePhysicsParams::build_models(VecModelBuilders const& model_builders,
                                        ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;
    models.reserve(model_builders.size());

    for (auto const& builder : model_builders)
    {
        auto action_id = action_reg.next_id();
        SPConstModel model = builder(action_id);

        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == action_id);

        action_reg.insert(model);
        models.push_back(std::move(model));
    }

    CELER_ENSURE(models.size() == model_builders.size());
    return models;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
