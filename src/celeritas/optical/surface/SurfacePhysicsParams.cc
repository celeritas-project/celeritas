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

namespace celeritas
{
namespace optical
{

template<class T>
class FakeModel : public SurfaceModel
{
  public:
    FakeModel(SurfaceModelId model_id,
              std::string_view label,
              std::map<inp::SurfaceLayer, T> const& layer_map)
        : SurfaceModel(model_id, label), layers_(layer_map)
    {
    }

    VecSurfaceLayer get_surfaces() const final
    {
        VecSurfaceLayer result;
        for ([[maybe_unused]] auto const& [layer, data] : layers_)
        {
            result.push_back(PhysSurfaceId(layer.get()));
        }
        return result;
    }

    void step(CoreParams const&, CoreStateHost&) const final {}
    void step(CoreParams const&, CoreStateDevice&) const final {}

  private:
    std::map<inp::SurfaceLayer, T> layers_;
};

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
    // Post boundary action
    {
        post_boundary_action_
            = std::make_shared<PostBoundaryAction>(action_reg->next_id());
        CELER_ASSERT(post_boundary_action_);
        action_reg->insert(post_boundary_action_);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    this->build_surfaces(input.materials, data);
    models_ = this->build_models(input, data);

    // Finalize data
    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Build sub-step surface physics models.
 */
auto SurfacePhysicsParams::build_models(
    inp::SurfacePhysics const& input,
    HostVal<SurfacePhysicsParamsData>& data) const
    -> SurfaceStepArray<std::vector<SPModel>>
{
    SurfaceStepArray<std::vector<SPModel>> step_models;

    {
        auto& roughness = step_models[SurfacePhysicsStep::roughness];

        if (!input.roughness.polished.empty())
        {
            roughness.push_back(std::make_shared<FakeModel<inp::NoRoughness>>(
                SurfaceModelId(roughness.size()),
                "polished",
                input.roughness.polished));
        }

        if (!input.roughness.smear.empty())
        {
            roughness.push_back(
                std::make_shared<FakeModel<inp::SmearRoughness>>(
                    SurfaceModelId(roughness.size()),
                    "smear",
                    input.roughness.smear));
        }

        if (!input.roughness.gaussian.empty())
        {
            roughness.push_back(
                std::make_shared<FakeModel<inp::GaussianRoughness>>(
                    SurfaceModelId(roughness.size()),
                    "gaussian",
                    input.roughness.gaussian));
        }
    }
    {
        auto& reflectivity = step_models[SurfacePhysicsStep::reflectivity];

        if (!input.reflectivity.grid.empty())
        {
            reflectivity.push_back(
                std::make_shared<FakeModel<inp::GridReflection>>(
                    SurfaceModelId(reflectivity.size()),
                    "grid",
                    input.reflectivity.grid));
        }

        if (!input.reflectivity.fresnel.empty())
        {
            reflectivity.push_back(
                std::make_shared<FakeModel<inp::FresnelReflection>>(
                    SurfaceModelId(reflectivity.size()),
                    "fresnel",
                    input.reflectivity.fresnel));
        }
    }
    {
        auto& interaction = step_models[SurfacePhysicsStep::interaction];

        if (!input.interaction.dielectric_dielectric.empty())
        {
            interaction.push_back(
                std::make_shared<FakeModel<inp::ReflectionForm>>(
                    SurfaceModelId(interaction.size()),
                    "dielectric-dielectric",
                    input.interaction.dielectric_dielectric));
        }

        if (!input.interaction.dielectric_metal.empty())
        {
            interaction.push_back(
                std::make_shared<FakeModel<inp::ReflectionForm>>(
                    SurfaceModelId(interaction.size()),
                    "dielectric-metal",
                    input.interaction.dielectric_metal));
        }
    }

    // Build surface physics maps
    for (auto step : range(SurfacePhysicsStep::size_))
    {
        SurfacePhysicsMapBuilder build_step(data.scalars.default_surface,
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
/*!
 * Build surface data form inputs.
 */
void SurfacePhysicsParams::build_surfaces(
    std::vector<std::vector<OptMatId>> const& interstitial_materials,
    HostVal<SurfacePhysicsParamsData>& data) const
{
    auto build_surface = make_builder(&data.surfaces);
    auto build_material = make_builder(&data.subsurface_materials);

    PhysSurfaceId next_phys_surface{0};
    for (auto const& materials : interstitial_materials)
    {
        PhysSurfaceId phys_surface_start = next_phys_surface;
        next_phys_surface
            = PhysSurfaceId(phys_surface_start.get() + materials.size() - 1);

        SurfaceRecord record{
            ItemMap<SubsurfaceMaterialId, OpaqueId<OptMatId>>{
                build_material.insert_back(materials.begin(), materials.end())},
            ItemMap<SubsurfaceInterfaceId, PhysSurfaceId>{
                range(phys_surface_start, next_phys_surface)}};

        build_surface.push_back(record);
    }

    data.scalars.default_surface = next_phys_surface;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
