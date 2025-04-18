//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/MockTestBase.cc
//---------------------------------------------------------------------------//
#include "MockTestBase.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "phys/MockProcess.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// PUBLIC MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
auto MockTestBase::make_applicability(char const* name,
                                      real_type lo_energy,
                                      real_type hi_energy) const -> Applicability
{
    CELER_EXPECT(name);
    CELER_EXPECT(lo_energy <= hi_energy);

    using units::MevEnergy;

    Applicability result;
    result.particle = this->particle()->find(name);
    result.lower = MevEnergy{lo_energy};
    result.upper = MevEnergy{hi_energy};
    return result;
}

//---------------------------------------------------------------------------//
auto MockTestBase::make_model_callback() const -> ModelCallback
{
    return [this](ActionId id) {
        CELER_ASSERT(id);
        interactions_.push_back(ModelId{id.unchecked_get() - model_to_action_});
    };
}

//---------------------------------------------------------------------------//
// PROTECTED MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
auto MockTestBase::build_material() -> SPConstMaterial
{
    using namespace units;
    MaterialParams::Input inp;
    inp.elements = {{AtomicNumber{1}, AmuMass{1.0}, {}, "celerogen"},
                    {AtomicNumber{4}, AmuMass{10.0}, {}, "celerinium"},
                    {AtomicNumber{15}, AmuMass{30.0}, {}, "celeron"}};
    inp.materials.push_back({native_value_from(InvCcDensity{1e20}),
                             300,
                             MatterState::gas,
                             {{ElementId{0}, 1.0}},
                             "lo density celerogen"});
    inp.materials.push_back({native_value_from(InvCcDensity{1e21}),
                             300,
                             MatterState::liquid,
                             {{ElementId{0}, 1.0}},
                             "hi density celerogen"});
    inp.materials.push_back(
        {native_value_from(InvCcDensity{1e23}),
         300,
         MatterState::solid,
         {{ElementId{0}, 0.1}, {ElementId{1}, 0.3}, {ElementId{2}, 0.6}},
         "celer composite"});
    inp.materials.push_back({native_value_from(InvCcDensity{1.0}),
                             2.7,
                             MatterState::gas,
                             {{ElementId{0}, 1.0}},
                             "the cold emptiness of space"});
    return std::make_shared<MaterialParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_geomaterial() -> SPConstGeoMaterial
{
    GeoMaterialParams::Input input;
    input.geometry = this->geometry();
    input.materials = this->material();
    input.volume_to_mat
        = {PhysMatId{0}, PhysMatId{2}, PhysMatId{1}, PhysMatId{3}};
    input.volume_labels
        = {Label{"inner"}, Label{"middle"}, Label{"outer"}, Label{"world"}};
    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_particle() -> SPConstParticle
{
    using namespace constants;
    using namespace units;
    constexpr auto zero = zero_quantity();

    ParticleParams::Input inp;
    inp.push_back({"gamma", pdg::gamma(), zero, zero, stable_decay_constant});
    inp.push_back({"celeriton",
                   PDGNumber{1337},
                   MevMass{1},
                   ElementaryCharge{1},
                   stable_decay_constant});
    inp.push_back({"anti-celeriton",
                   PDGNumber{-1337},
                   MevMass{1},
                   ElementaryCharge{-1},
                   stable_decay_constant});
    inp.push_back({"electron",
                   pdg::electron(),
                   MevMass{0.5109989461},
                   ElementaryCharge{-1},
                   stable_decay_constant});
    inp.push_back({"celerino",
                   PDGNumber{81},
                   MevMass{0},
                   ElementaryCharge{0},
                   stable_decay_constant});
    return std::make_shared<ParticleParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_cutoff() -> SPConstCutoff
{
    CutoffParams::Input input;
    input.materials = this->material();
    input.particles = this->particle();
    input.cutoffs = {};  // No cutoffs

    return std::make_shared<CutoffParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_physics() -> SPConstPhysics
{
    using Barn = MockProcess::BarnMicroXs;
    PhysicsParams::Input physics_inp;
    physics_inp.materials = this->material();
    physics_inp.particles = this->particle();
    physics_inp.options = this->build_physics_options();
    physics_inp.action_registry = this->action_reg().get();

    // Add a few processes
    auto interact = this->make_model_callback();
    {
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "scattering";
        inp.supports_integral_xs = false;
        inp.applic = {make_applicability("gamma", 1e-6, 100),
                      make_applicability("celeriton", 1, 100)};
        inp.xs = {Barn{1.0}, Barn{1.0}, Barn{1.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "absorption";
        inp.supports_integral_xs = false;
        inp.applic = {make_applicability("gamma", 1e-6, 100)};
        inp.xs = {Barn{2.0}, Barn{2.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Three different models for the single process
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "purrs";
        inp.applic = {make_applicability("celeriton", 1e-3, 1),
                      make_applicability("celeriton", 1, 10),
                      make_applicability("celeriton", 10, 100)};
        inp.xs = {Barn{3.0}, Barn{3.0}};
        inp.energy_loss = MevCmSqLossDens{0.6 * 1e-20};  // 0.6 MeV/cm in
                                                         // celerogen
        inp.interp = this->interpolation();
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Two models for anti-celeriton
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "hisses";
        inp.applies_at_rest = true;
        inp.applic = {make_applicability("anti-celeriton", 1e-3, 1),
                      make_applicability("anti-celeriton", 1, 100)};
        inp.xs = {Barn{4.0}, Barn{4.0}};
        inp.energy_loss = MevCmSqLossDens{0.7 * 1e-20};
        inp.interp = this->interpolation();
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "meows";
        inp.applic = {make_applicability("celeriton", 1e-3, 10),
                      make_applicability("anti-celeriton", 1e-3, 10)};
        inp.xs = {Barn{5.0}, Barn{5.0}};
        inp.energy_loss = {};
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    {
        // Energy-dependent cross section
        MockProcess::Input inp;
        inp.materials = this->material();
        inp.interact = interact;
        inp.label = "barks";
        inp.applic = {make_applicability("electron", 1e-5, 10)};
        inp.xs = {Barn{0}, Barn{6.0}, Barn{12.0}, Barn{6.0}};
        inp.energy_loss = MevCmSqLossDens{0.5 * 1e-20};
        inp.interp = this->interpolation();
        physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
    }
    return std::make_shared<PhysicsParams>(std::move(physics_inp));
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_along_step() -> SPConstAction
{
    auto& action_reg = *this->action_reg();
    auto result
        = AlongStepGeneralLinearAction::from_params(action_reg.next_id(),
                                                    *this->material(),
                                                    *this->particle(),
                                                    nullptr,
                                                    false);
    CELER_ASSERT(result);
    CELER_ASSERT(!result->has_fluct());
    CELER_ASSERT(!result->has_msc());
    action_reg.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_sim() -> SPConstSim
{
    SimParams::Input input;
    input.particles = this->particle();
    return std::make_shared<SimParams>(input);
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_init() -> SPConstTrackInit
{
    TrackInitParams::Input input;
    input.capacity = 4096;
    input.max_events = 4096;
    input.track_order = TrackOrder::none;
    return std::make_shared<TrackInitParams>(input);
}

//---------------------------------------------------------------------------//
auto MockTestBase::build_physics_options() const -> PhysicsOptions
{
    return {};
}

//---------------------------------------------------------------------------//
inp::Interpolation MockTestBase::interpolation() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
