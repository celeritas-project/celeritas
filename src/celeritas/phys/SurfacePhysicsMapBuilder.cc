//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapBuilder.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsMapBuilder.hh"

#include <utility>

#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Filler.hh"
#include "geocel/SurfaceParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with surface data and result to modify.
 */
SurfacePhysicsMapBuilder::SurfacePhysicsMapBuilder(SurfaceParams const& surfaces,
                                                   HostData& data)
    : surfaces_{surfaces}, data_{data}
{
    CELER_EXPECT(data_.action_ids.empty() && data_.model_surface_ids.empty());

    resize(&data_.action_ids, surfaces_.num_surfaces());
    fill(ActionId{}, &data_.action_ids);
    resize(&data_.model_surface_ids, surfaces_.num_surfaces());
    fill(SurfaceModel::ModelSurfaceId{}, &data_.model_surface_ids);
}

//---------------------------------------------------------------------------//
/*!
 * Add and index from a surface model.
 */
void SurfacePhysicsMapBuilder::operator()(SurfaceModel const& model)
{
    using ModelSurfaceId = SurfaceModel::ModelSurfaceId;

    auto action_id = model.action_id();
    {
        // Validate that the model's passed only once
        auto&& [iter, inserted] = actions_.insert(action_id);
        CELER_VALIDATE(inserted,
                       << "duplicate model " << model.label()
                       << " given to surface physics map builder");
        CELER_DISCARD(iter);
    }

    // TODO: this will need updating to support multiple layers
    ModelSurfaceId::size_type ms_index{0};
    for (SurfaceId surface_id : model.get_surfaces())
    {
        CELER_VALIDATE(surface_id < surfaces_.num_surfaces(),
                       << "surface physics model " << model.label()
                       << " contained invalid surface indices");

        // Assign and check the action ID
        ActionId prev_id
            = std::exchange(data_.action_ids[surface_id], model.action_id());
        CELER_VALIDATE(!prev_id,
                       << "multiple surface physics models were assigned to "
                          "the same surface");

        // Add the model surface ID
        data_.model_surface_ids[surface_id] = ModelSurfaceId{ms_index++};
    }
    CELER_VALIDATE(ms_index > 0,
                   << "surface physics model " << model.label()
                   << " is not associated with any surfaces");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
