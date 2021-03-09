//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.cc
//---------------------------------------------------------------------------//

#include "field/RungeKutta.hh"
#include "field/MagField.hh"
#include "field/FieldEquation.hh"
#include "field/base/OdeArray.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"
#include "base/Units.hh"
#include "physics/base/Units.hh"

#include "FieldTestParams.hh"
#include "celeritas_test.hh"

#ifdef CELERITAS_USE_CUDA
#    include "RungeKutta.test.hh"
#endif

using namespace celeritas;
using namespace celeritas_test;

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

        param.field_value = 1.0 * units::tesla; // field value along z [tesla]
        param.radius      = 38.085386036;       // radius of curvature [mm]
        param.delta_z     = 67.003310629;       // z-change/revolution [mm]
        param.momentum_y  = 10.9610028286;      // initial momentum_y  [MeV]
        param.momentum_z  = 3.1969591583;       // initial momentum_z  [MeV]
        param.nstates     = 32 * 512;           // number of states (tracks)
        param.nsteps      = 100;                // number of steps/revolution
        param.revolutions = 10;                 // number of revolutions
        param.epsilon     = 1.0e-5; // tolerance of the stepper error
    }

  protected:
    // Test parameters
    FieldTestParams param;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RungeKuttaTest, rk4_host)
{
    // Construct the Runge-Kutta stepper
    MagField      field({0, 0, param.field_value});
    FieldEquation equation(field);
    RungeKutta    rk4(equation);

    // Test parameters and the sub-step size
    real_type hstep = 2.0 * constants::pi * param.radius / param.nsteps;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(param.nstates))
    {
        // Initial state and the epected state after revolutions
        OdeArray<real_type, 6> y;
        y[0] = param.radius;
        y[2] = i * 1.0e-6;
        y[4] = param.momentum_y;
        y[5] = param.momentum_z;

        OdeArray<real_type, 6> expected_y = y;

        // The rhs of the equation and a temporary array
        OdeArray<real_type, 6> dydx;
        OdeArray<real_type, 6> yout;

        // Try the stepper by hstep for (num_revolutions * num_steps) times
        real_type total_err2 = 0;
        for (int nr = 0; nr < param.revolutions; ++nr)
        {
            // Travel hstep for num_steps times in the field
            expected_y[2] = param.delta_z * (nr + 1) + i * 1.0e-6;
            for (CELER_MAYBE_UNUSED int j : celeritas::range(param.nsteps))
            {
                dydx           = equation(y);
                yout           = rk4(hstep, y, dydx);
                real_type err2 = rk4.error(hstep, y);
                y              = yout;
                total_err2 += err2;
            }
            // Check the state after each revolution and the total error
            EXPECT_VEC_NEAR(expected_y.get(), y.get(), sqrt(total_err2));
            EXPECT_LT(total_err2, param.epsilon);
        }
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class RungeKuttaDeviceTest : public RungeKuttaTest
{
};

TEST_F(RungeKuttaDeviceTest, rk4_device)
{
    // Run kernel
    auto output = rk4_test(param);

    // Check stepper results
    real_type zstep = param.delta_z * param.revolutions;
    for (unsigned int i = 0; i < output.pos_x.size(); ++i)
    {
        real_type error = std::sqrt(output.error[i]);
        EXPECT_LT(fabs(output.pos_x[i] - param.radius), error);
        EXPECT_LT(fabs(output.pos_z[i] - (zstep + i * 1.0e-6)), error);
        EXPECT_LT(fabs(output.mom_y[i] - param.momentum_y), error);
        EXPECT_LT(fabs(output.mom_z[i] - param.momentum_z), error);
        EXPECT_LT(output.error[i], param.epsilon);
    }
}

//---------------------------------------------------------------------------//
#endif
