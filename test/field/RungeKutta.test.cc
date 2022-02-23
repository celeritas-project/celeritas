//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.cc
//---------------------------------------------------------------------------//

#include "field/RungeKuttaStepper.hh"
#include "field/UniformMagField.hh"
#include "field/MagFieldEquation.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"
#include "base/Units.hh"
#include "physics/base/Units.hh"

#include "FieldTestParams.hh"
#include "celeritas_test.hh"

#include "RungeKutta.test.hh"
#include "detail/MagTestTraits.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::detail::truncation_error;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RungeKuttaTest : public Test
{
  protected:
    void SetUp() override
    {
        /***
          Physical system of the test and parameters: the helix motion of
          an electron in a uniform magnetic field along the z-direction with
          initial velocity (v0), position (pos_0) and direction (dir_0).

          B     = {0.0, 0.0, 1.0*units::tesla};
          v_0   = 0.999*units::c_light
          pos_0 = {radius, 0.0, 0.0}
          dir_0 = {0.0, 0.96, 0.28}
        */

        param.field_value = 1.0 * units::tesla; //! field value along z [tesla]
        param.radius      = 3.8085386036;       //! radius of curvature [cm]
        param.delta_z     = 6.7003310629;       //! z-change/revolution [cm]
        param.momentum_y  = 10.9610028286;      //! initial momentum_y [MeV/c]
        param.momentum_z  = 3.1969591583;       //! initial momentum_z [MeV/c]
        param.nstates     = 32 * 512;           //! number of states (tracks)
        param.nsteps      = 100;                //! number of steps/revolution
        param.revolutions = 10;                 //! number of revolutions
        param.epsilon     = 1.0e-5;             //! tolerance error
    }

  protected:
    // Test parameters
    FieldTestParams param;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RungeKuttaTest, host)
{
    // Construct the Runge-Kutta stepper

    UniformMagField field({0, 0, param.field_value});

    using RKTraits = detail::MagTestTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);

    // Test parameters and the sub-step size
    real_type hstep = 2.0 * constants::pi * param.radius / param.nsteps;

    // Only test every 128 states to reduce debug runtime
    for (unsigned int i : celeritas::range(param.nstates).step(128u))
    {
        // Initial state and the epected state after revolutions
        OdeState y;
        y.pos = {param.radius, 0.0, i * 1.0e-6};
        y.mom = {0.0, param.momentum_y, param.momentum_z};

        OdeState expected_y = y;

        // Try the stepper by hstep for (num_revolutions * num_steps) times
        real_type total_err2 = 0;
        for (int nr : range(param.revolutions))
        {
            // Travel hstep for num_steps times in the field
            expected_y.pos[2] = param.delta_z * (nr + 1) + i * 1.0e-6;
            for (CELER_MAYBE_UNUSED int j : celeritas::range(param.nsteps))
            {
                StepperResult result = rk4(hstep, y);
                y                    = result.end_state;
                total_err2 += detail::truncation_error(
                    hstep, 0.001, y, result.err_state);
            }
            // Check the state after each revolution and the total error
            EXPECT_VEC_NEAR(expected_y.pos, y.pos, sqrt(total_err2));
            EXPECT_VEC_NEAR(expected_y.mom, y.mom, sqrt(total_err2));
            EXPECT_LT(total_err2, param.epsilon);
        }
    }
}

//---------------------------------------------------------------------------//

TEST_F(RungeKuttaTest, TEST_IF_CELER_DEVICE(device))
{
    // Run kernel
    auto output = rk4_test(param);

    // Check stepper results
    real_type zstep = param.delta_z * param.revolutions;
    for (auto i : celeritas::range(output.pos_x.size()))
    {
        real_type error = std::sqrt(output.error[i]);
        EXPECT_SOFT_NEAR(output.pos_x[i], param.radius, error);
        EXPECT_SOFT_NEAR(output.pos_z[i], zstep + i * 1.0e-6, error);
        EXPECT_SOFT_NEAR(output.mom_y[i], param.momentum_y, error);
        EXPECT_SOFT_NEAR(output.mom_z[i], param.momentum_z, error);
        EXPECT_LT(output.error[i], param.epsilon);
    }
}

//---------------------------------------------------------------------------//
