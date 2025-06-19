//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/surface/SurfaceModelBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GliSurModelBuilder : public SurfaceModelBuilder
{
  public:
    void build_facet_normal_actions(NormalActionBuilder&) const final;
    void
    build_calc_reflectivity_actions(ReflectivityActionBuilder&) const final;
    void build_interaction_actions(InteractionActionBuilder&) const final;
    SPModel build_surface_model(ActionId) const final;

  private:
    GliSurModel::Input input_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
