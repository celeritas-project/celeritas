//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/phys/SurfacePhysicsMapBuilder.hh"

#include "model/DielectricInteractionModel.hh"
#include "model/FresnelReflectivityModel.hh"
#include "model/GaussianRoughnessModel.hh"
#include "model/GridReflectivityModel.hh"
#include "model/PolishedRoughnessModel.hh"
#include "model/SmearRoughnessModel.hh"
#include "model/TrivialInteractionModel.hh"

#include "detail/BuiltinSurfaceModelBuilder.hh"

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Calculate number of physics surfaces as defined by interstitial materials.
 */
PhysSurfaceId::size_type
num_phys_surfaces(std::vector<std::vector<OptMatId>> const& materials)
{
    PhysSurfaceId::size_type num = 0;
    for (auto const& mats : materials)
    {
        num += mats.size() + 1;
    }
    return num;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct surface physics parameters from input.
 */
SurfacePhysicsParams::SurfacePhysicsParams(ActionRegistry* action_reg,
                                           inp::SurfacePhysics const& input)
{
    CELER_EXPECT(action_reg);

    // Build actions

    // Init boundary action
    {
        init_boundary_action_
            = std::make_shared<InitBoundaryAction>(action_reg->next_id());
        CELER_ASSERT(init_boundary_action_);
        action_reg->insert(init_boundary_action_);
    }
    // Surface stepping action
    {
        surface_stepping_action_
            = std::make_shared<SurfaceSteppingAction>(action_reg->next_id());
        CELER_ASSERT(surface_stepping_action_);
        action_reg->insert(surface_stepping_action_);
    }
    // Post boundary action
    {
        post_boundary_action_
            = std::make_shared<PostBoundaryAction>(action_reg->next_id());
        CELER_ASSERT(post_boundary_action_);
        action_reg->insert(post_boundary_action_);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    data.scalars.init_boundary_action = init_boundary_action_->action_id();
    data.scalars.post_boundary_action = post_boundary_action_->action_id();
    data.scalars.surface_stepping_action
        = surface_stepping_action_->action_id();

    this->build_surfaces(input.materials, data);
    models_ = this->build_models(input, data);

    // Finalize data
    CELER_ENSURE(data);

    data_ = ParamsDataStore<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Build surface data from inputs.
 */
void SurfacePhysicsParams::build_surfaces(
    std::vector<std::vector<OptMatId>> const& interstitial_materials,
    HostVal<SurfacePhysicsParamsData>& data) const
{
    CELER_EXPECT(!interstitial_materials.empty());

    auto build_surface = make_builder(&data.surfaces);
    auto build_material = make_builder(&data.subsurface_materials);

    PhysSurfaceId next_phys_surface{0};
    for (auto const& materials : interstitial_materials)
    {
        // Number of physics surfaces is equal to number of interstitial
        // materials plus one
        PhysSurfaceId phys_surface_start = next_phys_surface;
        next_phys_surface
            = PhysSurfaceId(phys_surface_start.get() + materials.size() + 1);

        build_surface.push_back(SurfaceRecord{
            build_material.insert_back(materials.begin(), materials.end()),
            range(phys_surface_start, next_phys_surface)});
    }

    // Default surface is last geometric surface ID
    data.scalars.default_surface = SurfaceId(interstitial_materials.size() - 1);
}

//---------------------------------------------------------------------------//
/*!
 * Build sub-step surface physics models.
 */
auto SurfacePhysicsParams::build_models(
    inp::SurfacePhysics const& input,
    HostVal<SurfacePhysicsParamsData>& data) const -> SurfaceStepModels
{
    SurfaceStepModels step_models;

    for (auto step : range(SurfacePhysicsOrder::size_))
    {
        // Build fake models
        detail::BuiltinSurfaceModelBuilder build_model{step_models[step]};
        switch (step)
        {
            case SurfacePhysicsOrder::roughness:
                build_model.build<PolishedRoughnessModel>(
                    input.roughness.polished);
                build_model.build<SmearRoughnessModel>(input.roughness.smear);
                build_model.build<GaussianRoughnessModel>(
                    input.roughness.gaussian);
                break;
            case SurfacePhysicsOrder::reflectivity:
                build_model.build<GridReflectivityModel>(
                    input.reflectivity.grid);
                build_model.build<FresnelReflectivityModel>(
                    input.reflectivity.fresnel);
                break;
            case SurfacePhysicsOrder::interaction:
                build_model.build<DielectricInteractionModel>(
                    input.interaction.dielectric);
                build_model.build<TrivialInteractionModel>(
                    input.interaction.trivial);
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }

        CELER_VALIDATE(
            build_model.num_surfaces() == num_phys_surfaces(input.materials),
            << "same number of physics surfaces required for each "
               "surface physics step ("
            << num_phys_surfaces(input.materials) << " expected surfaces, "
            << build_model.num_surfaces() << " surfaces from "
            << to_cstring(step) << " step)");

        // Build surface physics maps
        SurfacePhysicsMapBuilder build_step(build_model.num_surfaces(),
                                            data.model_maps[step]);
        for (auto const& model : step_models[step])
        {
            build_step(*model);
        }
        CELER_ENSURE(data.model_maps[step]);
    }

    return step_models;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
