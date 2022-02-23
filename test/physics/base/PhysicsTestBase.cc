//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTestBase.cc
//---------------------------------------------------------------------------//
#include "PhysicsTestBase.hh"

#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/material/MaterialParams.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
void PhysicsTestBase::SetUp()
{
    materials_ = this->build_materials();
    CELER_ASSERT(materials_);
    particles_ = this->build_particles();
    CELER_ASSERT(particles_);
    physics_ = this->build_physics();
    CELER_ASSERT(physics_);
}

//---------------------------------------------------------------------------//
auto PhysicsTestBase::build_materials() const -> SPConstMaterials
{
    using namespace celeritas::units;
    MaterialParams::Input inp;
    inp.elements
        = {{1, AmuMass{1.0}, "celerogen"}, {4, AmuMass{10.0}, "celerinium"}};
    inp.materials.push_back({1e20,
                             300,
                             MatterState::gas,
                             {{ElementId{0}, 1.0}},
                             "lo density celerogen"});
    inp.materials.push_back({1e21,
                             300,
                             MatterState::liquid,
                             {{ElementId{0}, 1.0}},
                             "hi density celerogen"});
    inp.materials.push_back({1e23,
                             300,
                             MatterState::solid,
                             {{ElementId{1}, 1.0}},
                             "solid celerinium"});
    return std::make_shared<MaterialParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
auto PhysicsTestBase::build_particles() const -> SPConstParticles
{
    using namespace celeritas::units;
    namespace pdg = celeritas::pdg;

    constexpr auto zero   = celeritas::zero_quantity();
    constexpr auto stable = celeritas::ParticleRecord::stable_decay_constant();

    ParticleParams::Input inp;
    inp.push_back({"gamma", pdg::gamma(), zero, zero, stable});
    inp.push_back(
        {"celeriton", PDGNumber{1337}, MevMass{1}, ElementaryCharge{1}, stable});
    inp.push_back({"anti-celeriton",
                   PDGNumber{-1337},
                   MevMass{1},
                   ElementaryCharge{-1},
                   stable});
    inp.push_back({"electron",
                   pdg::electron(),
                   MevMass{0.5109989461},
                   ElementaryCharge{-1},
                   stable});
    return std::make_shared<ParticleParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
auto PhysicsTestBase::build_physics_options() const -> PhysicsOptions
{
    return {};
}

//---------------------------------------------------------------------------//
auto PhysicsTestBase::build_physics() const -> SPConstPhysics
{
    using Barn = MockProcess::BarnMicroXs;
    PhysicsParams::Input physics_inp;
    physics_inp.materials = this->materials();
    physics_inp.particles = this->particles();
    physics_inp.options   = this->build_physics_options();

    // Add a few processes
    MockProcess::Input inp;
    inp.materials = this->materials();
    inp.interact  = this->make_model_callback();
    {
        inp.label       = "scattering";
        inp.type        = ProcessType::electromagnetic_discrete;
        inp.applic      = {make_applicability("gamma", 1e-6, 100),
                      make_applicability("celeriton", 1, 100)};
        inp.xs          = {Barn{1.0}, Barn{1.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        inp.label       = "absorption";
        inp.type        = ProcessType::electromagnetic_discrete;
        inp.applic      = {make_applicability("gamma", 1e-6, 100)};
        inp.xs          = {Barn{2.0}, Barn{2.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Three different models for the single process
        inp.label       = "purrs";
        inp.type        = ProcessType::electromagnetic_dedx;
        inp.applic      = {make_applicability("celeriton", 1e-3, 1),
                      make_applicability("celeriton", 1, 10),
                      make_applicability("celeriton", 10, 100)};
        inp.xs          = {Barn{3.0}, Barn{3.0}};
        inp.energy_loss = 0.6 * 1e-20; // 0.6 MeV/cm in celerogen
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Two models for anti-celeriton
        inp.label       = "hisses";
        inp.type        = ProcessType::electromagnetic_dedx;
        inp.applic      = {make_applicability("anti-celeriton", 1e-3, 1),
                      make_applicability("anti-celeriton", 1, 100)};
        inp.xs          = {Barn{4.0}, Barn{4.0}};
        inp.energy_loss = 0.7 * 1e-20;
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        inp.label       = "meows";
        inp.type        = ProcessType::electromagnetic_dedx;
        inp.applic      = {make_applicability("celeriton", 1e-3, 10),
                      make_applicability("anti-celeriton", 1e-3, 10)};
        inp.xs          = {Barn{5.0}, Barn{5.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Energy-dependent cross section
        inp.label       = "barks";
        inp.type        = ProcessType::electromagnetic_dedx;
        inp.applic      = {make_applicability("electron", 1e-5, 10)};
        inp.xs          = {Barn{0}, Barn{6.0}, Barn{12.0}, Barn{6.0}};
        inp.energy_loss = 0.5 * 1e-20;
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    return std::make_shared<PhysicsParams>(std::move(physics_inp));
}

//---------------------------------------------------------------------------//
auto PhysicsTestBase::make_applicability(const char* name,
                                         double      lo_energy,
                                         double      hi_energy) const
    -> Applicability
{
    using celeritas::units::MevEnergy;
    CELER_EXPECT(particles_);
    CELER_EXPECT(name);
    CELER_EXPECT(lo_energy <= hi_energy);
    Applicability result;
    result.particle = particles_->find(name);
    result.lower    = MevEnergy{lo_energy};
    result.upper    = MevEnergy{hi_energy};
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
