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
namespace
{
//---------------------------------------------------------------------------//
/*!
 * A fake model used to mock actual surface physics models.
 *
 * TODO: Remove and replace with actual built-in surface models.
 */
template<class T>
class FakeModel : public SurfaceModel
{
  public:
    FakeModel(SurfaceModelId model_id,
              std::string_view label,
              std::map<PhysSurfaceId, T> const& layer_map)
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
    std::map<PhysSurfaceId, T> layers_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper to build fake models.
 *
 * Doesn't insert empty models and keeps track of number of physics surfaces
 * for validation purposes.
 */
class FakeModelBuilder
{
  public:
    FakeModelBuilder(std::vector<std::shared_ptr<SurfaceModel>>& models)
        : models_(models)
    {
    }

    template<class T>
    void operator()(std::string_view label,
                    std::map<PhysSurfaceId, T> const& layer_map)
    {
        if (!layer_map.empty())
        {
            models_.push_back(std::make_shared<FakeModel<T>>(
                SurfaceModelId(models_.size()), label, layer_map));
            num_surf_ += layer_map.size();
        }
    }

    size_type num_surfaces() const { return num_surf_; }

  private:
    std::vector<std::shared_ptr<SurfaceModel>>& models_;
    size_type num_surf_{0};
};

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

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
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
        FakeModelBuilder build_model{step_models[step]};
        switch (step)
        {
            case SurfacePhysicsOrder::roughness:
                build_model("polished", input.roughness.polished);
                build_model("smear", input.roughness.smear);
                build_model("gaussian", input.roughness.gaussian);
                break;

            case SurfacePhysicsOrder::reflectivity:
                build_model("grid", input.reflectivity.grid);
                build_model("fresnel", input.reflectivity.fresnel);
                break;
            case SurfacePhysicsOrder::interaction:
                build_model("dielectric-dielectric",
                            input.interaction.dielectric_dielectric);
                build_model("dielectric-metal",
                            input.interaction.dielectric_metal);
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }

        CELER_VALIDATE(
            build_model.num_surfaces() == num_phys_surfaces(input.materials),
            << " same number of physics surfaces required for each "
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
