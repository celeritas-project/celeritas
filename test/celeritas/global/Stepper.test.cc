//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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

#include "../SimpleTestBase.hh"
#include "../TestEm15Base.hh"
#include "../TestEm3Base.hh"
#include "DummyAction.hh"
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
    make_primaries_with_energy(size_type count, MevEnergy energy) const
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::electron());
        CELER_ASSERT(p.particle_id);
        p.energy = energy;
        p.track_id = TrackId{0};
        p.position = {-22, 0, 0};
        p.direction = {1, 0, 0};
        p.time = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
#define TestEm3NoMsc TEST_IF_CELERITAS_GEANT(TestEm3NoMsc)
class TestEm3NoMsc : public TestEm3StepperTestBase
{
  public:
    //! Make 10GeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10000});
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
#define TestEm15MscField TEST_IF_CELERITAS_GEANT(TestEm15MscField)
class TestEm15MscField : public TestEm15Base, public StepperTestBase
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
            action_reg.next_id(), field_params, msc);
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
// TESTEM3
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
        "pre-step",
        "along-step-general-linear",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-combined",
        "geo-boundary",
        "dummy-action",
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
        EXPECT_EQ(331, result.num_step_iters());
        EXPECT_SOFT_EQ(62216, result.calc_avg_steps_per_primary());
        EXPECT_EQ(256, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({95, 1182}), result.calc_queue_hwm());
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
    EXPECT_EQ(24, counts.active);
    EXPECT_EQ(24, counts.alive);

    // Transport existing tracks
    counts = step();
    EXPECT_EQ(44, counts.active);
    EXPECT_EQ(44, counts.alive);
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
        EXPECT_EQ(206, result.num_step_iters());
        EXPECT_SOFT_EQ(62127.625, result.calc_avg_steps_per_primary());
        EXPECT_EQ(72, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({70, 908}), result.calc_queue_hwm());
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
        "pre-step",
        "along-step-general-linear",
        "physics-discrete-select",
        "scat-klein-nishina",
        "photoel-livermore",
        "conv-bethe-heitler",
        "annihil-2-gamma",
        "ioni-moller-bhabha",
        "brems-combined",
        "geo-boundary",
        "dummy-action",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(TestEm3Msc, host)
{
    size_type num_primaries = 8;
    size_type num_tracks = 2048;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(45.125, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        EXPECT_EQ(67, result.num_step_iters());
        EXPECT_SOFT_EQ(44, result.calc_avg_steps_per_primary());
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
        EXPECT_EQ(60, result.num_step_iters());
        EXPECT_SOFT_EQ(45.75, result.calc_avg_steps_per_primary());
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
    EXPECT_SOFT_NEAR(58, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        if (CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ(85, result.num_step_iters());
            EXPECT_SOFT_EQ(63.25, result.calc_avg_steps_per_primary());
        }
        else
        {
            EXPECT_EQ(79, result.num_step_iters());
            EXPECT_SOFT_EQ(61.125, result.calc_avg_steps_per_primary());
        }
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
        if (CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ(94, result.num_step_iters());
            EXPECT_SOFT_EQ(64, result.calc_avg_steps_per_primary());
        }
        else
        {
            EXPECT_EQ(92, result.num_step_iters());
            EXPECT_SOFT_EQ(63.75, result.calc_avg_steps_per_primary());
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

TEST_F(TestEm15MscField, setup)
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
        "pre-step",
        "along-step-uniform-msc",
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
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(TestEm15MscField, host)
{
    size_type num_primaries = 4;
    size_type num_tracks = 1024;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(35, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        EXPECT_EQ(14, result.num_step_iters());
        EXPECT_SOFT_EQ(34.5, result.calc_avg_steps_per_primary());
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

TEST_F(TestEm15MscField, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 8;
    size_type num_tracks = 1024;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    if (this->is_ci_build())
    {
        EXPECT_EQ(16, result.num_step_iters());
        EXPECT_SOFT_EQ(30, result.calc_avg_steps_per_primary());
        EXPECT_EQ(5, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({2, 11}), result.calc_queue_hwm());
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
