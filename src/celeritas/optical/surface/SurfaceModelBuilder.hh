//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
enum class FacetNormalActionClass
{
    trivial,
    glisur_polished,
    unified,
    size_
};

enum class CalculateReflectivityActionClass
{
    from_grid,
    fresnel_dielectric,
    fresnel_metal,
    size_
};

enum class InteractionActionClass
{
    unified_dielectric_metal,
    unified_dielectric_dielectric,
    size_
};

//---------------------------------------------------------------------------//
/*!
 */
class SurfaceModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    using NormalActionBuilder = SharedActionSetBuilder<FacetNormalActionClass>;
    using ReflectivityActionBuilder
        = SharedActionSetBuilder<CalculateReflectivityActionClass>;
    using InteractionActionBuilder
        = SharedActionSetBuilder<InteractionActionClass>;
    //!@}

  public:
    //! Build facet normal actions
    virtual void build_facet_normal_actions(NormalActionBuilder&) const = 0;

    //! Build calculate reflectivity actions
    virtual void
    build_calc_reflectivity_actions(ReflectivityActionBuilder&) const
        = 0;

    //! Build interaction actions
    virtual void build_interaction_actions(InteractionActionBuilder&) const = 0;

    //! Build surface model
    virtual SPModel build_surface_model(ActionId) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
