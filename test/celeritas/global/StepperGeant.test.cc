//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperGeant.test.cc
//---------------------------------------------------------------------------//
#include <random>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/LeadBoxTestBase.hh"
#include "celeritas/OneSteelSphereBase.hh"
#include "celeritas/TestEm15Base.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/field/UniformFieldParams.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Primary.hh"

#include "StepperTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TestEm3StepperTestBase : public TestEm3Base, public StepperTestBase
{
  public:
    std::vector<Primary>
    make_primaries_with_energy(PDGNumber particle,
                               size_type count,
                               celeritas::units::MevEnergy energy) const
    {
        Primary p;
        p.particle_id = this->particle()->find(particle);
        CELER_ASSERT(p.particle_id);
        p.energy = energy;
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }

    // Return electron primaries as default
    std::vector<Primary>
    make_primaries_with_energy(size_type count,
                               celeritas::units::MevEnergy energy) const
    {
        return this->make_primaries_with_energy(pdg::electron(), count, energy);
    }
};

//---------------------------------------------------------------------------//
#define TestEm3Compton TEST_IF_CELERITAS_GEANT(TestEm3Compton)
class TestEm3Compton : public TestEm3StepperTestBase
{
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(
            pdg::gamma(), count, units::MevEnergy{1});
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.compton_scattering = true;
        opts.coulomb_scattering = false;
        opts.photoelectric = false;
        opts.rayleigh_scattering = false;
        opts.gamma_conversion = false;
        opts.gamma_general = false;
        opts.ionization = false;
        opts.annihilation = false;
        opts.brems = BremsModelSelection::none;
        opts.msc = MscModelSelection::none;
        opts.relaxation = RelaxationSelection::none;
        opts.lpm = false;
        opts.eloss_fluctuation = false;
        return opts;
    }

    size_type max_average_steps() const override { return 1000; }
};

//---------------------------------------------------------------------------//
#define TestEm3NoMsc TEST_IF_CELERITAS_GEANT(TestEm3NoMsc)
class TestEm3NoMsc : public TestEm3StepperTestBase
{
  public:
    //! Make 10GeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(
            count, celeritas::units::MevEnergy{10000});
    }

    size_type max_average_steps() const override
    {
        return 100000;  // 8 primaries -> ~500k steps, be conservative
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.msc = MscModelSelection::none;
        return opts;
    }
};

//---------------------------------------------------------------------------//

#define TestEm3Msc TEST_IF_CELERITAS_GEANT(TestEm3Msc)
class TestEm3Msc : public TestEm3StepperTestBase
{
  public:
    //! Make 10MeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10});
    }

    size_type max_average_steps() const override { return 100; }
};

//---------------------------------------------------------------------------//
#define TestEm3MscNofluct TEST_IF_CELERITAS_GEANT(TestEm3MscNofluct)
class TestEm3MscNofluct : public TestEm3Msc
{
  public:
    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.eloss_fluctuation = false;
        return opts;
    }
};

//---------------------------------------------------------------------------//
#define TestEm15FieldMsc TEST_IF_CELERITAS_GEANT(TestEm15FieldMsc)
class TestEm15FieldMsc : public TestEm15Base, public StepperTestBase
{
    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm15Base::build_geant_options();
        opts.eloss_fluctuation = false;
        return opts;
    }

    SPConstAction build_along_step() override
    {
        auto& action_reg = *this->action_reg();
        UniformFieldParams::Input field_inp;
        field_inp.strength = {0, 0, 1e-3};

        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        CELER_ASSERT(msc);

        auto result = std::make_shared<AlongStepUniformMscAction>(
            action_reg.next_id(), *this->geometry(), field_inp, nullptr, msc);
        action_reg.insert(result);
        return result;
    }

    //! Make isotropic 10MeV electron/positron mix
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = MevEnergy{10};
        p.position = {0, 0, 0};
        p.time = 0;

        Array<ParticleId, 2> const particles = {
            this->particle()->find(pdg::electron()),
            this->particle()->find(pdg::positron()),
        };
        CELER_ASSERT(particles[0] && particles[1]);

        std::vector<Primary> result(count, p);
        IsotropicDistribution<> sample_dir;
        std::mt19937 rng;

        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
            result[i].direction = sample_dir(rng);
            result[i].particle_id = particles[i % particles.size()];
        }
        return result;
    }

    size_type max_average_steps() const override { return 500; }
};

//---------------------------------------------------------------------------//
#define TestEm3MscNoIntegral TEST_IF_CELERITAS_GEANT(TestEm3MscNoIntegral)
class TestEm3MscNoIntegral : public TestEm3Msc
{
  public:
    //! Make 10 MeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10});
    }

    PhysicsOptions build_physics_options() const override
    {
        auto opts = ImportedDataTestBase::build_physics_options();
        opts.disable_integral_xs = true;
        return opts;
    }
};

//---------------------------------------------------------------------------//
#define OneSteelSphere TEST_IF_CELERITAS_GEANT(OneSteelSphere)
class OneSteelSphere : public OneSteelSphereBase, public StepperTestBase
{
    //! Make isotropic 10MeV electron/positron/gamma mix
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = MevEnergy{10};
        p.position = {0, 0, 0};
        p.time = 0;
        p.event_id = EventId{0};

        Array<ParticleId, 3> const particles = {
            this->particle()->find(pdg::gamma()),
            this->particle()->find(pdg::electron()),
            this->particle()->find(pdg::positron()),
        };
        CELER_ASSERT(particles[0] && particles[1] && particles[2]);

        std::vector<Primary> result(count, p);
        IsotropicDistribution<> sample_dir;
        std::mt19937 rng;

        for (auto i : range(count))
        {
            result[i].direction = sample_dir(rng);
            result[i].particle_id = particles[i % particles.size()];
        }
        return result;
    }

    size_type max_average_steps() const override { return 500; }
};

//---------------------------------------------------------------------------//
#define LeadBox TEST_IF_CELERITAS_GEANT(LeadBox)
class LeadBox : public LeadBoxTestBase, public StepperTestBase
{
    //! Make electron that fails to change position after propagation
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::electron());
        p.energy = MevEnergy{1};
        p.position = {1e20, 0, 0};
        p.direction = {-1, 0, 0};
        p.time = 0;
        p.event_id = EventId{0};

        std::vector<Primary> result(count, p);
        return result;
    }

    size_type max_average_steps() const override { return 500; }
};

//---------------------------------------------------------------------------//
// TESTEM3 - Compton process only
//---------------------------------------------------------------------------//

TEST_F(TestEm3Compton, setup)
{
    auto result = this->check_setup();
    static char const* expected_process[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_process, result.processes);
}

TEST_F(TestEm3Compton, host)
{
    size_type num_primaries = 1;
    size_type num_tracks = 256;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_EQ(153, result.num_step_iters());
        EXPECT_SOFT_EQ(796, result.calc_avg_steps_per_primary());
        EXPECT_EQ(47, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({6, 1}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM3 - No MSC
//---------------------------------------------------------------------------//

TEST_F(TestEm3NoMsc, setup)
{
    auto result = this->check_setup();

    static char const* const expected_processes[] = {
        "Compton scattering",
        "Photoelectric effect",
        "Photon annihiliation",
        "Positron annihiliation",
        "Electron/positron ionization",
        "Bremsstrahlung",
    };
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "pre-step",
        "along-step-general-linear",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-sb",
        "brems-rel",
        "geo-boundary",
        "tracking-cut",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(TestEm3NoMsc, host)
{
    size_type num_primaries = 1;
    size_type num_tracks = 256;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(63490, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        EXPECT_EQ(329, result.num_step_iters());
        EXPECT_SOFT_EQ(59335, result.calc_avg_steps_per_primary());
        EXPECT_EQ(225, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({133, 1355}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm3NoMsc, host_multi)
{
    // Run and inject multiple sets of primaries during transport

    size_type num_primaries = 8;
    size_type num_tracks = 128;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    auto counts = step(make_span(primaries));
    EXPECT_EQ(num_primaries, counts.active);
    EXPECT_EQ(num_primaries, counts.alive);

    // Transport existing tracks
    counts = step();
    EXPECT_EQ(num_primaries, counts.active);
    EXPECT_EQ(num_primaries, counts.alive);

    // Add some more primaries
    primaries = this->make_primaries(num_primaries);
    counts = step(make_span(primaries));
    if (this->is_default_build())
    {
        EXPECT_EQ(24, counts.active);
        EXPECT_EQ(24, counts.alive);
    }

    // Transport existing tracks
    counts = step();
    if (this->is_default_build())
    {
        EXPECT_EQ(36, counts.active);
        EXPECT_EQ(35, counts.alive);
    }
}

TEST_F(TestEm3NoMsc, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 8;
    // Num tracks is low enough to hit capacity
    size_type num_tracks = num_primaries * 800;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(62756.625, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        EXPECT_EQ(212, result.num_step_iters());
        EXPECT_SOFT_EQ(61803.25, result.calc_avg_steps_per_primary());
        EXPECT_EQ(87, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({75, 1888}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM3_MSC
//---------------------------------------------------------------------------//

TEST_F(TestEm3Msc, setup)
{
    auto result = this->check_setup();

    static char const* const expected_processes[] = {
        "Compton scattering",
        "Photoelectric effect",
        "Photon annihiliation",
        "Positron annihiliation",
        "Electron/positron ionization",
        "Bremsstrahlung",
    };
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "pre-step",
        "along-step-general-linear",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-sb",
        "brems-rel",
        "geo-boundary",
        "tracking-cut",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);

    static char const* const expected_actions_desc[] = {
        "create track initializers from primaries",
        "initialize track states",
        "update beginning-of-step state",
        "apply along-step for particles with no field",
        "apply along-step for neutral particles",
        "select a discrete interaction",
        "interact by Compton scattering (simple Klein-Nishina)",
        "interact by Livermore photoelectric effect",
        "interact by Bethe-Heitler gamma conversion",
        "interact by positron annihilation yielding two gammas",
        "interact by Moller+Bhabha ionization",
        "interact by Seltzer-Berger bremsstrahlung",
        "interact by relativistic bremsstrahlung",
        "cross a geometry boundary",
        "kill a track and deposit its energy",
        "create track initializers from secondaries",
    };
    EXPECT_VEC_EQ(expected_actions_desc, result.actions_desc);
}

TEST_F(TestEm3Msc, host)
{
    size_type num_primaries = 8;
    size_type num_tracks = 2048;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_EQ(58, result.num_step_iters());
        EXPECT_LE(37.375, result.calc_avg_steps_per_primary());
        EXPECT_GE(40, result.calc_avg_steps_per_primary());
        EXPECT_EQ(10, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({8, 6}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm3Msc, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 8;
    size_type num_tracks = 1024;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_EQ(60, result.num_step_iters());
        EXPECT_SOFT_EQ(43.625, result.calc_avg_steps_per_primary());
        EXPECT_EQ(9, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({5, 6}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM3_MSC_NOFLUCT
//---------------------------------------------------------------------------//

TEST_F(TestEm3MscNofluct, host)
{
    size_type num_primaries = 8;
    size_type num_tracks = 2048;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_LE(69, result.num_step_iters());
        EXPECT_GE(73, result.num_step_iters());
        EXPECT_LE(58.625, result.calc_avg_steps_per_primary());
        EXPECT_GE(63.125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(8, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({4, 5}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm3MscNofluct, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 8;
    size_type num_tracks = 1024;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_LE(76, result.num_step_iters());
        EXPECT_GE(77, result.num_step_iters());
        EXPECT_LE(48, result.calc_avg_steps_per_primary());
        EXPECT_GE(48.25, result.calc_avg_steps_per_primary());
        EXPECT_EQ(7, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({5, 7}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM15_MSC_NOINTEGRAL
//---------------------------------------------------------------------------//

TEST_F(TestEm3MscNoIntegral, host)
{
    size_type num_primaries = 24;
    size_type num_tracks = 2048;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_LE(86, result.num_step_iters());
        EXPECT_GE(87, result.num_step_iters());
        EXPECT_LE(54.7, result.calc_avg_steps_per_primary());
        EXPECT_GE(54.75, result.calc_avg_steps_per_primary());
        EXPECT_EQ(8, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({6, 15}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// TESTEM15_MSC_FIELD
//---------------------------------------------------------------------------//

TEST_F(TestEm15FieldMsc, setup)
{
    auto result = this->check_setup();

    static char const* const expected_processes[] = {
        "Compton scattering",
        "Photoelectric effect",
        "Photon annihiliation",
        "Positron annihiliation",
        "Electron/positron ionization",
        "Bremsstrahlung",
    };
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "pre-step",
        "along-step-uniform-msc",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-sb",
        "brems-rel",
        "geo-boundary",
        "tracking-cut",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(TestEm15FieldMsc, host)
{
    size_type num_primaries = 4;
    size_type num_tracks = 1024;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_ci_build())
    {
        EXPECT_EQ(15, result.num_step_iters());
        EXPECT_SOFT_EQ(38, result.calc_avg_steps_per_primary());
        EXPECT_EQ(6, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({4, 7}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm15FieldMsc, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 8;
    size_type num_tracks = 1024;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    if (this->is_ci_build())
    {
        EXPECT_EQ(14, result.num_step_iters());
        EXPECT_SOFT_EQ(34.125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(5, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({4, 12}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

//---------------------------------------------------------------------------//
// ONESTEELSPHERE
//---------------------------------------------------------------------------//

TEST_F(OneSteelSphere, setup)
{
    auto result = this->check_setup();

    static char const* const expected_processes[] = {
        "Compton scattering",
        "Photoelectric effect",
        "Photon annihiliation",
        "Positron annihiliation",
        "Electron/positron ionization",
        "Bremsstrahlung",
    };
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "pre-step",
        "along-step-general-linear",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-sb",
        "brems-rel",
        "geo-boundary",
        "tracking-cut",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(OneSteelSphere, host)
{
    size_type num_primaries = 128;
    size_type num_tracks = 1024;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(15.8671875, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        EXPECT_EQ(16, result.num_step_iters());
        EXPECT_SOFT_EQ(16.265625, result.calc_avg_steps_per_primary());
        EXPECT_EQ(7, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({4, 116}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(LeadBox, host)
{
    size_type num_primaries = 1;
    size_type num_tracks = 8;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    // Electron fails to change position in first step and is killed by
    // tracking cut
    EXPECT_EQ(1, result.calc_avg_steps_per_primary());
    EXPECT_EQ(1, result.num_step_iters());
    EXPECT_EQ(0, result.calc_emptying_step());
    EXPECT_EQ(RunResult::StepCount({0, 0}), result.calc_queue_hwm());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
