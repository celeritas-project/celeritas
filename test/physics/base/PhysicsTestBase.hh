//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "gtest/Test.hh"

#include <random>
#include "physics/material/MaterialParams.hh"
#include "physics/base/ParticleParams.hh"

// Test helpers
#include "MockProcess.hh"
#include "random/DiagnosticRngEngine.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;
class PhysicsParams;
} // namespace celeritas

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
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    using SPConstMaterials = std::shared_ptr<celeritas::MaterialParams>;
    using SPConstParticles = std::shared_ptr<celeritas::ParticleParams>;
    using SPConstPhysics   = std::shared_ptr<celeritas::PhysicsParams>;
    using Applicability    = celeritas::Applicability;
    using ModelId          = celeritas::ModelId;
    using ModelCallback    = std::function<void(ModelId)>;
    //!@}

  protected:
    virtual ~PhysicsTestBase();

    void SetUp() override;

    virtual SPConstMaterials build_materials() const;
    virtual SPConstParticles build_particles() const;
    virtual SPConstPhysics   build_physics() const;

    const SPConstMaterials& materials() const { return materials_; }
    const SPConstParticles& particles() const { return particles_; }
    const SPConstPhysics&   physics() const { return physics_; }

  public:
    Applicability make_applicability(const char* name,
                                     double      lo_energy,
                                     double      hi_energy) const;

    ModelCallback make_model_callback() const
    {
        return [this](ModelId id) { interactions_.push_back(id); };
    }

    celeritas::Span<const ModelId> called_models() const
    {
        return make_span(interactions_);
    }

    //!@{
    //! Random number generator
    RandomEngine& rng() { return rng_; }
    //!@}


  private:
    //// DATA ////

    SPConstMaterials materials_;
    SPConstParticles particles_;
    SPConstPhysics   physics_;
    RandomEngine     rng_;

    mutable std::vector<ModelId> interactions_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
