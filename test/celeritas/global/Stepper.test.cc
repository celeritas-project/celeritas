//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/Stepper.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "DummyAction.hh"
#include "StepperTestBase.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

#define TestEm3Test TEST_IF_CELERITAS_GEANT(TestEm3Test)
class TestEm3Test : public celeritas_test::TestEm3Base,
                    public celeritas_test::StepperTestBase
{
  public:
    //! Make 10GeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10000});
    }

    size_type max_average_steps() const override
    {
        return 100000; // 8 primaries -> ~500k steps, be conservative
    }

    std::vector<Primary>
    make_primaries_with_energy(size_type count, MevEnergy energy) const
    {
        Primary p;
        p.particle_id = this->particle()->find("e-");
        CELER_ASSERT(p.particle_id);
        p.energy    = energy;
        p.track_id  = TrackId{0};
        p.position  = {-22, 0, 0};
        p.direction = {1, 0, 0};
        p.time      = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// XXX: disabled due to MSC errors
// #define TestEm3MscTest TEST_IF_CELERITAS_GEANT(TestEm3MscTest)
#define TestEm3MscTest DISABLED_TestEm3MscTest
class TestEm3MscTest : public TestEm3Test
{
  public:
    //! Use MSC
    bool enable_msc() const override { return true; }

    //! Make 10MeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10});
    }
};

//---------------------------------------------------------------------------//
// XXX: disabled due to MSC errors
//#define TestEm3MscNofluctTest TEST_IF_CELERITAS_GEANT(TestEm3MscNofluctTest)
#define TestEm3MscNofluctTest DISABLED_TestEm3MscNofluctTest
class TestEm3MscNofluctTest : public TestEm3Test
{
  public:
    //! Use MSC
    bool enable_msc() const override { return true; }
    //! Disable energy loss fluctuation
    bool enable_fluctuation() const override { return false; }

    //! Make 10MeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(count, MevEnergy{10});
    }
};

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    size_type num_primaries   = 1;
    size_type inits_per_track = 32 * 8;
    size_type num_tracks      = num_primaries * inits_per_track;

    Stepper<MemSpace::host> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(58000, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_ci_build())
    {
        result.print_expected();
        FAIL() << "Add CI data here";
    }
    else if (this->is_wildstyle_build())
    {
        EXPECT_EQ(343, result.num_step_iters());
        EXPECT_SOFT_EQ(63490, result.calc_avg_steps_per_primary());
        EXPECT_EQ(255, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({108, 1416}), result.calc_queue_hwm());
    }
    else if (this->is_srj_build())
    {
        EXPECT_EQ(321, result.num_step_iters());
        EXPECT_SOFT_EQ(54526, result.calc_avg_steps_per_primary());
        EXPECT_EQ(207, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({90, 1268}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
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

TEST_F(TestEm3Test, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 1024;
    // Num tracks is low enough to hit capacity
    size_type num_tracks = num_primaries * 800;

    Stepper<MemSpace::device> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);
    EXPECT_SOFT_NEAR(58000, result.calc_avg_steps_per_primary(), 0.10);

    if (this->is_wildstyle_build())
    {
        EXPECT_EQ(218, result.num_step_iters());
        EXPECT_SOFT_EQ(62756.625, result.calc_avg_steps_per_primary());
        EXPECT_EQ(82, result.calc_emptying_step());
        EXPECT_EQ(RunResult::StepCount({75, 1450}), result.calc_queue_hwm());
    }
    else
    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
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

TEST_F(TestEm3MscTest, host)
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 32 * 8;
    size_type num_tracks      = num_primaries * inits_per_track;

    Stepper<MemSpace::host> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);

    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm3MscTest, device)
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 512;
    size_type num_tracks      = 1024;

    Stepper<MemSpace::device> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);

    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
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

TEST_F(TestEm3MscNofluctTest, host)
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 32 * 8;
    size_type num_tracks      = num_primaries * inits_per_track;

    Stepper<MemSpace::host> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);

    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}

TEST_F(TestEm3MscNofluctTest, device)
{
    size_type num_primaries   = 8;
    size_type inits_per_track = 512;
    size_type num_tracks      = 1024;

    Stepper<MemSpace::device> step(
        this->make_stepper_input(num_tracks, inits_per_track));
    auto result = this->run(step, num_primaries);

    {
        cout << "No output saved for combination of "
             << celeritas_test::PrintableBuildConf{} << std::endl;
        result.print_expected();

        if (this->strict_testing())
        {
            FAIL() << "Updated stepper results are required for CI tests";
        }
    }
}
