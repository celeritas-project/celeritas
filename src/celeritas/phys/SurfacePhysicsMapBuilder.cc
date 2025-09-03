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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with surface data and result to modify.
 */
SurfacePhysicsMapBuilder::SurfacePhysicsMapBuilder(
    PhysSurfaceId::size_type num_surfaces, HostData& data)
    : data_{data}
{
    CELER_EXPECT(data_.surface_models.empty()
                 && data_.internal_surface_ids.empty());

    resize(&data_.surface_models, num_surfaces);
    resize(&data_.internal_surface_ids, num_surfaces);

    fill(SurfaceModelId{}, &data_.surface_models);
    fill(InternalSurfaceId{}, &data_.internal_surface_ids);

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Add and index from a surface model.
 */
void SurfacePhysicsMapBuilder::operator()(SurfaceModel const& model)
{
    auto surface_model_id = model.surface_model_id();
    {
        // Validate that the model's passed only once
        auto&& [iter, inserted] = surface_models_.insert(surface_model_id);
        CELER_VALIDATE(inserted,
                       << "duplicate model " << model.label()
                       << " given to surface physics map builder");
        CELER_DISCARD(iter);
    }

    InternalSurfaceId::size_type ms_index{0};
    for (PhysSurfaceId surface_id : model.get_surfaces())
    {
        CELER_VALIDATE(surface_id < data_.surface_models.size(),
                       << "surface physics model " << model.label()
                       << " contained invalid surface indices");

        // Assign and check the model ID
        SurfaceModelId prev_id = std::exchange(
            data_.surface_models[surface_id], model.surface_model_id());
        CELER_VALIDATE(!prev_id,
                       << "multiple surface physics models were assigned to "
                          "the same surface");

        // Add the model surface ID
        data_.internal_surface_ids[surface_id] = InternalSurfaceId{ms_index++};
    }
    CELER_VALIDATE(ms_index > 0,
                   << "surface physics model " << model.label()
                   << " is not associated with any surfaces");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
