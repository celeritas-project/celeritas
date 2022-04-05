//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <random>

#include "physics/base/PhysicsParams.hh"
#include "MockProcess.hh"
#include "gtest/Test.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class ActionManager;
}

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Construct mock materials, particles, and physics data.
 *
 * This class creates two elements, three single-element materials, three
 * particles, and five MockProcesses, each of which emits one or more
 * MockModels.
 * - gamma:scattering
 * - gamma:absorption
 * - celeriton:scattering
 * - celeriton:purrs
 * - celeriton:meows
 * - anti-celeriton:hisses
 * - anti-celeriton:meows
 */
class PhysicsTestBase : public celeritas::Test
{
  protected:
    //!@{
    //! Type aliases

    using SPMaterials = std::shared_ptr<celeritas::MaterialParams>;
    using SPParticles = std::shared_ptr<celeritas::ParticleParams>;
    using SPPhysics   = std::shared_ptr<celeritas::PhysicsParams>;
    using SPActionManager  = std::shared_ptr<celeritas::ActionManager>;
    using PhysicsOptions   = celeritas::PhysicsParams::Options;
    using Applicability    = celeritas::Applicability;
    using ActionId         = celeritas::ActionId;
    using ModelCallback    = std::function<void(ActionId)>;
    using ModelId         = celeritas::ModelId;
    //!@}

  protected:
    void SetUp() override;

    virtual SPMaterials    build_materials() const;
    virtual SPParticles    build_particles() const;
    virtual PhysicsOptions build_physics_options() const;
    virtual SPPhysics      build_physics() const;

    const SPMaterials& materials() const { return materials_; }
    const SPParticles& particles() const { return particles_; }
    const SPPhysics&   physics() const { return physics_; }

  public:
    ~PhysicsTestBase();

    Applicability make_applicability(const char* name,
                                     double      lo_energy,
                                     double      hi_energy) const;

    ModelCallback make_model_callback() const;

    celeritas::Span<const ModelId> called_models() const
    {
        return make_span(interactions_);
    }

  private:
    //// DATA ////

    SPActionManager actions_;
    SPMaterials materials_;
    SPParticles particles_;
    SPPhysics   physics_;

    mutable std::vector<ModelId> interactions_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
