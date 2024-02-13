//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/Stepper.hh"

#include <random>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"

#include "StepperTestBase.hh"
#include "celeritas_test.hh"
#include "../OneSteelSphereBase.hh"
#include "../SimpleTestBase.hh"
#include "../TestEm15Base.hh"
#include "../TestEm3Base.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SimpleComptonTest : public SimpleTestBase, public StepperTestBase
{
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy = units::MevEnergy{100};
        p.track_id = TrackId{0};
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

    size_type max_average_steps() const override { return 100000; }
};

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
        p.track_id = TrackId{0};
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
        UniformFieldParams field_params;
        field_params.field = {0, 0, 1e-3 * units::tesla};

        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        CELER_ASSERT(msc);

        auto result = std::make_shared<AlongStepUniformMscAction>(
            action_reg.next_id(), field_params, nullptr, msc);
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
        p.track_id = TrackId{0};

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
            result[i].track_id = TrackId{i};
            result[i].direction = sample_dir(rng);
            result[i].particle_id = particles[i % particles.size()];
        }
        return result;
    }

    size_type max_average_steps() const override { return 500; }
};

//---------------------------------------------------------------------------//
// Two boxes: compton with fake cross sections
//---------------------------------------------------------------------------//

TEST_F(SimpleComptonTest, setup)
{
    auto result = this->check_setup();
    static char const* expected_process[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_process, result.processes);
}

TEST_F(SimpleComptonTest, host)
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

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
        "brems-combined",
        "geo-boundary",
        "dummy-action",
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
        EXPECT_EQ(326, result.num_step_iters());
        EXPECT_SOFT_EQ(61146, result.calc_avg_steps_per_primary());
        EXPECT_EQ(247, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({93, 1182}), result.calc_queue_hwm());
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

    // Check that callback was called
    EXPECT_EQ(result.active.size(), this->dummy_action().num_execute_host());
    EXPECT_EQ(0, this->dummy_action().num_execute_device());
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
        EXPECT_EQ(44, counts.active);
        EXPECT_EQ(44, counts.alive);
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
        EXPECT_EQ(195, result.num_step_iters());
        EXPECT_SOFT_EQ(62448.375, result.calc_avg_steps_per_primary());
        EXPECT_EQ(72, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({69, 883}), result.calc_queue_hwm());
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

    // Check that callback was called
    EXPECT_EQ(result.active.size(), this->dummy_action().num_execute_device());
    EXPECT_EQ(0, this->dummy_action().num_execute_host());
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
        "brems-combined",
        "geo-boundary",
        "dummy-action",
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
        "interact by bremsstrahlung (combined SB/relativistic, e+/-)",
        "cross a geometry boundary",
        "count the number of executions",
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
        EXPECT_EQ(86, result.num_step_iters());
        EXPECT_LE(46, result.calc_avg_steps_per_primary());
        EXPECT_GE(46.125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(10, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({1, 4}), result.calc_queue_hwm());
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
        EXPECT_EQ(64, result.num_step_iters());
        EXPECT_SOFT_EQ(CELERITAS_USE_VECGEOM ? 44.5 : 44.375,
                       result.calc_avg_steps_per_primary());
        EXPECT_EQ(8, result.calc_emptying_step());
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
        EXPECT_LE(70, result.num_step_iters());
        EXPECT_GE(73, result.num_step_iters());
        EXPECT_LE(61.625, result.calc_avg_steps_per_primary());
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
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
        {
            EXPECT_EQ(66, result.num_step_iters());
            EXPECT_SOFT_EQ(56.125, result.calc_avg_steps_per_primary());
        }
        else
        {
            EXPECT_EQ(64, result.num_step_iters());
            EXPECT_SOFT_EQ(52.5, result.calc_avg_steps_per_primary());
        }

        EXPECT_EQ(7, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({5, 8}), result.calc_queue_hwm());
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
        "dummy-action",
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
        EXPECT_EQ(16, result.num_step_iters());
        EXPECT_SOFT_EQ(35.25, result.calc_avg_steps_per_primary());
        EXPECT_EQ(5, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({2, 7}), result.calc_queue_hwm());
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
        EXPECT_EQ(17, result.num_step_iters());
        EXPECT_SOFT_EQ(34, result.calc_avg_steps_per_primary());
        EXPECT_EQ(5, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({1, 10}), result.calc_queue_hwm());
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
        "dummy-action",
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
        EXPECT_EQ(18, result.num_step_iters());
        EXPECT_SOFT_EQ(16.578125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(7, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({5, 119}), result.calc_queue_hwm());
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
}  // namespace test
}  // namespace celeritas
