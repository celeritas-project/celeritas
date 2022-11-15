//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/MockTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <vector>

#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"

#include "GlobalGeoTestBase.hh"

namespace celeritas
{
struct Applicability;
struct PhysicsParamsOptions;
} // namespace celeritas

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Three concentric spheres with mock physics.
 *
 * This class creates three elements, two single-element materials and one
 * multi-element material, four particles, and six MockProcesses, each of which
 * emits one or more MockModels.
 * - gamma:scattering
 * - gamma:absorption
 * - celeriton:scattering
 * - celeriton:purrs
 * - celeriton:meows
 * - anti-celeriton:hisses
 * - anti-celeriton:meows
 * - electron:barks
 *
 * Cutoff values are all zero.
 */
class MockTestBase : virtual public GlobalGeoTestBase
{
  public:
    //!@{
    //! Type aliases
    using PhysicsOptions = PhysicsParamsOptions;
    using ModelCallback  = std::function<void(ActionId)>;
    using SpanConstModel = Span<const ModelId>;
    //!@}

  public:
    Applicability make_applicability(const char* name,
                                     double      lo_energy,
                                     double      hi_energy) const;

    ModelCallback make_model_callback() const;

    inline Span<const ModelId> called_models() const
    {
        return make_span(interactions_);
    }

  protected:
    const char* geometry_basename() const override { return "three-spheres"; }

    SPConstMaterial    build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle    build_particle() override;
    SPConstCutoff      build_cutoff() override;
    SPConstPhysics     build_physics() override;
    SPConstAction      build_along_step() override;
    SPConstTrackInit   build_init() override { CELER_ASSERT_UNREACHABLE(); }

    virtual PhysicsOptions build_physics_options() const;

  private:
    //// DATA ////

    mutable std::vector<ModelId> interactions_;
    ActionId::size_type          model_to_action_{0};
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
