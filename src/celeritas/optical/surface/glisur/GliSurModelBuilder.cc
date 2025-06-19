//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurModelBuilder.cc
//---------------------------------------------------------------------------//
#include "GliSurModelBuilder.hh"

#include "celeritas/optical/surface/detail/GridReflectivityAction.hh"
#include "celeritas/optical/surface/detail/TrivialFacetNormalAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

void GliSurModelBuilder::build_facet_normal_actions(NormalActionBuilder& build) const
{
    input_.trivial_normal_action
        = build(FacetNormalActionClass::trivial, [](ActionId aid) {
              return std::make_shared<TrivialFacetNormalAction>(aid);
          });
    input_.glisur_normal_action
        = build(FacetNormalActionClass::glisur_polished, [](ActionId aid) {
              return std::make_shared<GliSurPolishedNormalAction>(aid);
          });
}

void GliSurModelBuilder::build_calc_reflectivity_actions(
    ReflectivityActionBuilder& build) const
{
    input_.grid_reflectivity_action
        = build(CalculateReflectivityActionClass::from_grid, [](ActionId aid) {
              return std::make_shared<GridReflectivityAction>(aid);
          });
}

void GliSurModelBuilder::build_interaction_actions(
    InteractionActionBuilder& build) const
{
    input_.glisur_dielectric_interaction = build(
        InteractionActionClass::unified_dielectric_dielectric, [](ActionId aid) {
            return std::make_shared<UnifiedDielectricDielectricAction>(aid);
        });
    input_.glisur_metal_interaction = build(
        InteractionActionClass::unified_dielectric_metal, [](ActionId aid) {
            return std::make_shared<UnifiedDielectricMetal>(aid);
        });
}

auto GliSurModelBuilder::build_surface_model(ActionId aid) const -> SPModel
{
    return std::make_shared<GliSurModel>(aid, input_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
