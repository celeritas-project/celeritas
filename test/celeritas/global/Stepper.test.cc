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
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "StepperTestBase.hh"
#include "celeritas_test.hh"
#include "../SimpleTestBase.hh"

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
}  // namespace test
}  // namespace celeritas
