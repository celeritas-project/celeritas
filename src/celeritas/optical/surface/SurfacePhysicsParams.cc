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

    data.scalars.init_boundary_action = init_boundary_action_->action_id();
    data.scalars.post_boundary_action = post_boundary_action_->action_id();

    this->build_surfaces(input.materials, data);
    models_ = this->build_models(input, data);

    // Finalize data
    CELER_ENSURE(data.scalars);
    CELER_ENSURE(!data.surfaces.empty());
    for (auto const& model_map : data.model_maps)
    {
        CELER_ENSURE(model_map);
    }
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

    SurfaceStepArray<size_type> num_surfaces{0, 0, 0};
    {
        auto& roughness = step_models[SurfacePhysicsStep::roughness];
        auto& num_rough_surf = num_surfaces[SurfacePhysicsStep::roughness];

        if (!input.roughness.polished.empty())
        {
            roughness.push_back(std::make_shared<FakeModel<inp::NoRoughness>>(
                SurfaceModelId(roughness.size()),
                "polished",
                input.roughness.polished));

            num_rough_surf += input.roughness.polished.size();
        }

        if (!input.roughness.smear.empty())
        {
            roughness.push_back(
                std::make_shared<FakeModel<inp::SmearRoughness>>(
                    SurfaceModelId(roughness.size()),
                    "smear",
                    input.roughness.smear));

            num_rough_surf += input.roughness.smear.size();
        }

        if (!input.roughness.gaussian.empty())
        {
            roughness.push_back(
                std::make_shared<FakeModel<inp::GaussianRoughness>>(
                    SurfaceModelId(roughness.size()),
                    "gaussian",
                    input.roughness.gaussian));

            num_rough_surf += input.roughness.gaussian.size();
        }
    }
    {
        auto& reflectivity = step_models[SurfacePhysicsStep::reflectivity];
        auto& num_refl_surf = num_surfaces[SurfacePhysicsStep::reflectivity];

        if (!input.reflectivity.grid.empty())
        {
            reflectivity.push_back(
                std::make_shared<FakeModel<inp::GridReflection>>(
                    SurfaceModelId(reflectivity.size()),
                    "grid",
                    input.reflectivity.grid));

            num_refl_surf += input.reflectivity.grid.size();
        }

        if (!input.reflectivity.fresnel.empty())
        {
            reflectivity.push_back(
                std::make_shared<FakeModel<inp::FresnelReflection>>(
                    SurfaceModelId(reflectivity.size()),
                    "fresnel",
                    input.reflectivity.fresnel));

            num_refl_surf += input.reflectivity.fresnel.size();
        }
    }
    {
        auto& interaction = step_models[SurfacePhysicsStep::interaction];
        auto& num_int_surf = num_surfaces[SurfacePhysicsStep::interaction];

        if (!input.interaction.dielectric_dielectric.empty())
        {
            interaction.push_back(
                std::make_shared<FakeModel<inp::ReflectionForm>>(
                    SurfaceModelId(interaction.size()),
                    "dielectric-dielectric",
                    input.interaction.dielectric_dielectric));

            num_int_surf += input.interaction.dielectric_dielectric.size();
        }

        if (!input.interaction.dielectric_metal.empty())
        {
            interaction.push_back(
                std::make_shared<FakeModel<inp::ReflectionForm>>(
                    SurfaceModelId(interaction.size()),
                    "dielectric-metal",
                    input.interaction.dielectric_metal));

            num_int_surf += input.interaction.dielectric_metal.size();
        }
    }

    CELER_VALIDATE(
        num_surfaces[SurfacePhysicsStep::roughness]
                == num_surfaces[SurfacePhysicsStep::reflectivity]
            && num_surfaces[SurfacePhysicsStep::roughness]
                   == num_surfaces[SurfacePhysicsStep::interaction],
        << " same number of surfaces required for each surface physics step ("
        << num_surfaces[SurfacePhysicsStep::roughness] << " roughness, "
        << num_surfaces[SurfacePhysicsStep::reflectivity] << " reflectivity, "
        << num_surfaces[SurfacePhysicsStep::interaction] << " interaction)");

    // Build surface physics maps
    for (auto step : range(SurfacePhysicsStep::size_))
    {
        SurfacePhysicsMapBuilder build_step(num_surfaces[step],
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
    CELER_EXPECT(!interstitial_materials.empty());

    auto build_surface = make_builder(&data.surfaces);
    auto build_material = make_builder(&data.subsurface_materials);

    PhysSurfaceId next_phys_surface{0};
    for (auto surface_id : range(interstitial_materials.size() - 1))
    {
        auto const& materials = interstitial_materials[surface_id];
        PhysSurfaceId phys_surface_start = next_phys_surface;
        next_phys_surface
            = PhysSurfaceId(phys_surface_start.get() + materials.size() + 1);

        SurfaceRecord record{
            ItemMap<SubsurfaceMaterialId, OpaqueId<OptMatId>>{
                build_material.insert_back(materials.begin(), materials.end())},
            ItemMap<SubsurfaceInterfaceId, PhysSurfaceId>{
                range(phys_surface_start, next_phys_surface)}};

        build_surface.push_back(record);
    }

    // Construct default surface
    data.scalars.default_surface = SurfaceId(interstitial_materials.size() - 1);
    auto const& default_materials = interstitial_materials.back();

    build_surface.push_back(
        SurfaceRecord{ItemMap<SubsurfaceMaterialId, OpaqueId<OptMatId>>{
                          build_material.insert_back(default_materials.begin(),
                                                     default_materials.end())},
                      ItemMap<SubsurfaceInterfaceId, PhysSurfaceId>{range(
                          next_phys_surface,
                          next_phys_surface + default_materials.size() + 1)}});
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
