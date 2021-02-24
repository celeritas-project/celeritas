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
          Physical Parameters and constant values for stepper testings:
          an electron entering perpendicular to a uniform magnetic field

          B = 1.0 tesla = 1E-3 (MeV/eplus)*(ns/mm*mm)
          v_0 = 0.999*c

          c = 299.792458 mm/ns
          M = 0.510998910 MeV
          m = M/c^2

          gamma = 22.3662720421293741
          K = 10.9181415106 MeV

          V = sqrt(K*K + 2.0 * M * K) = 11.417711279751803 [Mev/c]
          R = (M/c^2) * gamma * v_0/B = 38.08538603615336  [mm]
          T = 2*pi / (B/((M/c^2) * gamma))
        */

        params.nstates     = 128 * 128;
        params.nsteps      = 100;
        params.revolutions = 10;
        params.field_value = 0.001;
        params.radius      = 38.085386;
        params.momentum    = 11.417711;
        params.epsilon     = 1.0e-5; // tolerance of the steppererror
    }

  protected:
    // Test parameters
    FieldTestParams params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RungeKuttaTest, rk4_host)
{
    // Construct the Runge-Kutta stepper
    MagField      field({0, 0, params.field_value});
    FieldEquation equation(field);
    RungeKutta    rk4(equation);

    // Test parameters and the sub-step size
    real_type hstep = 2.0 * constants::pi * params.radius / params.nsteps;

    for (auto i : celeritas::range(params.nstates))
    {
        // Initial state and the epected state after revolutions
        OdeArray y({params.radius, 0, i * 1.0e-6, 0, params.momentum, 0});
        OdeArray expected_y = y;

        // The rhs of the equation and a temporary array
        OdeArray dydx;
        OdeArray yout;

        // Try the stepper by hstep for (num_revolutions * num_steps) times
        real_type total_error = 0;
        for (int nr = 0; nr < params.revolutions; ++nr)
        {
            // Travel hstep for num_steps times in the field
            for (CELER_MAYBE_UNUSED int j : celeritas::range(params.nsteps))
            {
                equation(y, dydx);
                real_type error = rk4.stepper(hstep, y, dydx, yout);
                y               = yout;
                total_error += error;
            }
            // Check the state after each revolution and the total error
            EXPECT_VEC_NEAR(expected_y.get(), y.get(), sqrt(total_error));
            EXPECT_LT(total_error, params.epsilon);
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
    auto output = rk4_test(params);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos_x.size(); ++i)
    {
        EXPECT_LT(fabs(output.pos_x[i] - params.radius), params.epsilon);
        EXPECT_LT(fabs(output.mom_y[i] - params.momentum), params.epsilon);
        EXPECT_LT(output.error[i], params.epsilon);
    }
}

//---------------------------------------------------------------------------//
#endif
