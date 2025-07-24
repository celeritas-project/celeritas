//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ActionRegistry.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the surface physics parameters from surface inputs.
 *
 * Creates actions in the following order:
 * - InitBoundaryAction
 * - Roughness models
 * - Reflectivity models
 * - Interaction models
 * - PostBoundaryAction
 */
SurfacePhysicsParams::SurfacePhysicsParams(Input input)
{
    CELER_EXPECT(input.action_registry);

    auto& action_reg = *input.action_registry;

    // Init boundary action
    {
        init_boundary_action_
            = std::make_shared<InitBoundaryAction>(action_reg.next_id());
        CELER_ASSERT(init_boundary_action_);
        action_reg.insert(init_boundary_action_);
    }

    // Register roughness actions
    // Register reflectivity actions
    // Register interaction actions

    // Post boundary action
    {
        post_boundary_action_
            = std::make_shared<PostBoundaryAction>(action_reg.next_id());
        CELER_ASSERT(post_boundary_action_);
        action_reg.insert(post_boundary_action_);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    // Construct scalars

    // Construct surfaces
    auto surface_builder = make_builder(&data.surfaces);
    for (auto n : input.num_subsurface_interfaces)
    {
        SurfacePhysicsRecord r;
        r.subsurface_materials
            = ItemMap<SubsurfaceMaterialId, SubsurfaceMaterialRecordId>(
                range(SubsurfaceMaterialRecordId{n + 1}));
        r.subsurface_interfaces
            = ItemMap<SubsurfaceInterfaceId, SubsurfaceInterfaceRecordId>(
                range(SubsurfaceInterfaceRecordId{n}));
        surface_builder.push_back(r);
    }

    // Finalize data

    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
