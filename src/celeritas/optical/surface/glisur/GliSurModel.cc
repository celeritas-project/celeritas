//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurModel.cc
//---------------------------------------------------------------------------//
#include "GliSurModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

GliSurModel::GliSurModel(ActionId aid, Input const& input)
    : SurfaceModel(aid, "glisur-model", "GliSur optical surface model")
{
    HostValue data;

    // Initialize scalars
    {
        auto& scalars = data.scalars;

        scalars.trivial_normal_action = input.trivial_normal_action;
        scalars.glisur_normal_action = input.glisur_normal_action;
        scalars.grid_reflectivity_action = input.grid_reflectivity_action;
        scalars.glisur_dielectric_interaction
            = input.glisur_dielectric_interaction;
        scalars.glisur_metal_interaction = input.glisur_metal_interaction;

        CELER_ASSERT(scalars);
    }

    HostValue<GliSurPolishedNormalData> polish_data;

    for (auto const& surface : input.surfaces)
    {
        CELER_EXPECT(surface);

        polish_data.polish.push_back(surface.polish);
        data.finish.push_back(surface.finish);
        data.interface_type.push_back(surface.interface_type);
    }

    CELER_ASSERT(polish_data.polish.size() == input.surfaces.size());
    CELER_ASSERT(data.finish.size() == input.surfaces.size());
    CELER_ASSERT(data.interface_type.size() == input.surfaces.size());

    CELER_ENSURE(data);
    CELER_ENSURE(polish_data);

    data_ = CollectionMirror<GliSurData>{std::move(data)};
    glisur_polished_normal_data_
        = CollectionMirror<GliSurPolishedNormalData>(polish_data);
}

void GliSurModel::step(CoreParams const& params, CoreStateHost& state) const {}

#if !CELER_USE_DEVICE
void GliSurModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
