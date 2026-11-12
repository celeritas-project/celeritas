//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/DecayProcess.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/decay/DecayProcess.hh"

#include <random>

#include "corecel/data/StateDataStore.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/MockTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/decay/channel/MuDecayChannel.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/MockProcess.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Muons with decay, plus a scattering and an at-rest process for mu-.
 *
 * - mu-: scattering, capture (at rest), decay
 * - mu+: decay
 */
class DecayProcessTest : public MockTestBase
{
    using Base = MockTestBase;
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

  protected:
    using MaterialStateStore
        = StateDataStore<MaterialStateData, MemSpace::host>;
    using ParticleStateStore
        = StateDataStore<ParticleStateData, MemSpace::host>;
    using PhysicsStateStore = StateDataStore<PhysicsStateData, MemSpace::host>;
    using MevEnergy = units::MevEnergy;
    using Barn = MockProcess::BarnMicroXs;

    static constexpr double muon_lifetime_s = 2.1969811e-6;

    SPConstParticle build_particle() override
    {
        using namespace constants;
        using namespace units;
        constexpr auto zero = zero_quantity();
        real_type const muon_decay_constant
            = 1 / (muon_lifetime_s * units::second);

        ParticleParams::Input inp;
        inp.push_back(
            {"gamma", pdg::gamma(), zero, zero, stable_decay_constant});
        inp.push_back({"electron",
                       pdg::electron(),
                       MevMass{0.5109989461},
                       ElementaryCharge{-1},
                       stable_decay_constant});
        inp.push_back({"positron",
                       pdg::positron(),
                       MevMass{0.5109989461},
                       ElementaryCharge{1},
                       stable_decay_constant});
        inp.push_back({"mu-",
                       pdg::mu_minus(),
                       MevMass{105.6583755},
                       ElementaryCharge{-1},
                       muon_decay_constant});
        inp.push_back({"mu+",
                       pdg::mu_plus(),
                       MevMass{105.6583755},
                       ElementaryCharge{1},
                       muon_decay_constant});
        return std::make_shared<ParticleParams>(std::move(inp));
    }

    SPConstPhysics build_physics() override
    {
        PhysicsParams::Input physics_inp;
        physics_inp.materials = this->material();
        physics_inp.particles = this->particle();
        physics_inp.options = this->build_physics_options();
        physics_inp.action_registry = this->action_reg().get();

        auto interact = this->make_model_callback();
        {
            MockProcess::Input inp;
            inp.materials = this->material();
            inp.interact = interact;
            inp.label = "scattering";
            inp.supports_integral_xs = false;
            inp.applic = {make_applicability("mu-", 1e-3, 1e3)};
            inp.xs = {Barn{1.0}, Barn{1.0}};
            physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
        }
        {
            MockProcess::Input inp;
            inp.materials = this->material();
            inp.interact = interact;
            inp.label = "capture";
            inp.supports_integral_xs = false;
            inp.applies_at_rest = true;
            inp.applic = {make_applicability("mu-", 1e-3, 1e3)};
            inp.xs = {Barn{1.0}, Barn{1.0}};
            physics_inp.processes.push_back(std::make_shared<MockProcess>(inp));
        }

        inp::DecayPhysics decay;
        decay.muon.parents = {pdg::mu_minus(), pdg::mu_plus()};
        physics_inp.decay
            = std::make_shared<DecayProcess>(this->particle(), decay);

        return std::make_shared<PhysicsParams>(std::move(physics_inp));
    }

    void SetUp() override
    {
        Base::SetUp();
        mat_state = MaterialStateStore(this->material()->host_ref(), 1);
        par_state = ParticleStateStore(this->particle()->host_ref(), 1);
        phys_state = PhysicsStateStore(this->physics()->host_ref(), 1);
    }

    //! Initialize a single track and return its physics view
    PhysicsTrackView init_track(char const* name, MevEnergy energy)
    {
        material_view() = MaterialTrackView::Initializer_t{PhysMatId{0}};

        ParticleTrackView::Initializer_t par_init;
        par_init.particle_id = this->particle()->find(name);
        CELER_EXPECT(par_init.particle_id);
        par_init.energy = energy;
        particle_view() = par_init;

        PhysicsTrackView phys(this->physics()->host_ref(),
                              phys_state.ref(),
                              particle_view(),
                              PhysMatId{0},
                              TrackSlotId{0});
        phys = PhysicsTrackView::Initializer_t{};
        return phys;
    }

    MaterialTrackView material_view()
    {
        return {this->material()->host_ref(), mat_state.ref(), TrackSlotId{0}};
    }
    ParticleTrackView particle_view()
    {
        return {this->particle()->host_ref(), par_state.ref(), TrackSlotId{0}};
    }
    PhysicsStepView step_view()
    {
        return {this->physics()->host_ref(), phys_state.ref(), TrackSlotId{0}};
    }

    //! Calculate the step limit, then force and select a discrete interaction
    ActionId select_interaction(PhysicsTrackView& phys)
    {
        auto material = this->material_view();
        auto particle = this->particle_view();
        auto pstep = this->step_view();
        phys.interaction_mfp(1);
        calc_physics_step_limit(material, particle, phys, pstep);

        // Testing cheat: force a discrete interaction
        PhysicsTrackView::PhysicsStateRef state_shortcut(phys_state.ref());
        state_shortcut.state[TrackSlotId{0}].interaction_mfp = 0;

        MaterialView mat_view(this->material()->host_ref(), PhysMatId{0});
        return select_discrete_interaction(
            mat_view, particle, phys, pstep, rng_);
    }

    //! Expected decay length [cm] of a muon with the given kinetic energy
    static double muon_decay_length_cm(double ke)
    {
        double const mass = 105.6583755;
        double const momentum = std::sqrt(ke * (ke + 2 * mass));
        double const c_cm_per_s = 2.99792458e10;
        return momentum / mass * c_cm_per_s * muon_lifetime_s;
    }

    MaterialStateStore mat_state;
    ParticleStateStore par_state;
    PhysicsStateStore phys_state;
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DecayProcessTest, params)
{
    auto const& physics = *this->physics();
    auto const& particles = *this->particle();

    // Decay is added after the interaction processes
    ASSERT_TRUE(physics.decay_process());
    EXPECT_EQ("Decay", physics.decay_process()->label());
    EXPECT_EQ(2, physics.num_processes());
    ProcessId const decay_id{2};
    EXPECT_EQ(decay_id, physics.host_ref().scalars.decay);

    // One channel for both muons
    ASSERT_EQ(1, physics.decay_channels().size());
    auto const& channel = *physics.decay_channels().front();
    EXPECT_EQ("decay-muon", channel.label());
    auto mu_minus = particles.find(pdg::mu_minus());
    auto mu_plus = particles.find(pdg::mu_plus());
    EXPECT_EQ((std::vector<ParticleId>{mu_minus, mu_plus}), channel.parents());

    // Channel actions are registered after the failure action
    EXPECT_LT(physics.host_ref().scalars.failure_action(), channel.action_id());

    auto const& groups = physics.host_ref().process_groups;

    // mu-: the immediate at-rest process takes priority over decay
    {
        auto const& group = groups[mu_minus];
        EXPECT_EQ(
            (std::vector<ProcessId>{ProcessId{0}, ProcessId{1}, decay_id}),
            std::vector<ProcessId>(physics.processes(mu_minus).begin(),
                                   physics.processes(mu_minus).end()));
        EXPECT_EQ(ParticleProcessId{1}, group.at_rest);
        EXPECT_EQ(channel.action_id(), group.decay_action);
    }
    // mu+: decay is the only process and applies at rest
    {
        auto const& group = groups[mu_plus];
        EXPECT_EQ((std::vector<ProcessId>{decay_id}),
                  std::vector<ProcessId>(physics.processes(mu_plus).begin(),
                                         physics.processes(mu_plus).end()));
        EXPECT_EQ(ParticleProcessId{0}, group.at_rest);
        EXPECT_EQ(channel.action_id(), group.decay_action);
    }
    // Stable particles don't decay
    EXPECT_FALSE(groups[particles.find(pdg::electron())].decay_action);
}

//---------------------------------------------------------------------------//
TEST_F(DecayProcessTest, step_limit)
{
    for (real_type ke : {real_type{1}, real_type{100}})
    {
        SCOPED_TRACE(ke);
        PhysicsTrackView phys = this->init_track("mu+", MevEnergy{ke});
        phys.interaction_mfp(1);
        auto material = this->material_view();
        auto particle = this->particle_view();
        auto pstep = this->step_view();
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);

        // Only decay limits the mu+ step (no energy loss)
        double expected = muon_decay_length_cm(ke);
        EXPECT_SOFT_EQ(expected, to_cm(particle.decay_length()));
        EXPECT_SOFT_EQ(1 / expected,
                       native_value_to<units::InvCmXs>(
                           pstep.per_process_xs(ParticleProcessId{0}))
                           .value());
        EXPECT_SOFT_EQ(expected, to_cm(step.step));
        EXPECT_EQ(this->physics()->host_ref().scalars.discrete_action(),
                  step.action);
    }

    // Decay competes with scattering for mu-
    {
        PhysicsTrackView phys = this->init_track("mu-", MevEnergy{100});
        phys.interaction_mfp(1);
        auto material = this->material_view();
        auto particle = this->particle_view();
        auto pstep = this->step_view();
        calc_physics_step_limit(material, particle, phys, pstep);
        real_type decay_xs = pstep.per_process_xs(ParticleProcessId{2});
        EXPECT_SOFT_EQ(1 / muon_decay_length_cm(100),
                       native_value_to<units::InvCmXs>(decay_xs).value());
        EXPECT_SOFT_EQ(pstep.per_process_xs(ParticleProcessId{0})
                           + pstep.per_process_xs(ParticleProcessId{1})
                           + decay_xs,
                       pstep.macro_xs());
    }

    // A stopped muon has no decay cross section: it decays at rest
    {
        PhysicsTrackView phys = this->init_track("mu+", zero_quantity());
        phys.interaction_mfp(1);
        auto material = this->material_view();
        auto particle = this->particle_view();
        auto pstep = this->step_view();
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(0, pstep.per_process_xs(ParticleProcessId{0}));
        EXPECT_EQ(0, step.step);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DecayProcessTest, select)
{
    ActionId const decay_action
        = this->physics()->decay_channels().front()->action_id();

    // In flight: mu+ can only decay
    {
        PhysicsTrackView phys = this->init_track("mu+", MevEnergy{10});
        EXPECT_EQ(decay_action, this->select_interaction(phys));
    }
    // At rest: mu+ decays
    {
        PhysicsTrackView phys = this->init_track("mu+", zero_quantity());
        EXPECT_EQ(decay_action, this->select_interaction(phys));
    }
    // At rest: mu- is captured rather than decaying (the mock capture model
    // has no data at zero energy, so check the process selection only)
    {
        PhysicsTrackView phys = this->init_track("mu-", zero_quantity());
        phys.interaction_mfp(1);
        auto material = this->material_view();
        auto particle = this->particle_view();
        auto pstep = this->step_view();
        calc_physics_step_limit(material, particle, phys, pstep);

        MaterialView mat_view(this->material()->host_ref(), PhysMatId{0});
        auto ppid = find_ppid(mat_view, particle, phys, pstep, rng_);
        EXPECT_EQ(ParticleProcessId{1}, ppid);
        EXPECT_EQ(ProcessId{1}, phys.process(ppid));
    }
}

//---------------------------------------------------------------------------//
TEST_F(DecayProcessTest, invalid_input)
{
    auto const& particles = *this->particle();
    inp::MuonDecay muon;

    // No parents
    EXPECT_THROW(MuDecayChannel(ActionId{0}, particles, muon), RuntimeError);

    // Not a muon
    muon.parents = {pdg::electron()};
    EXPECT_THROW(MuDecayChannel(ActionId{0}, particles, muon), RuntimeError);

    // Duplicate
    muon.parents = {pdg::mu_plus(), pdg::mu_plus()};
    EXPECT_THROW(MuDecayChannel(ActionId{0}, particles, muon), RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
