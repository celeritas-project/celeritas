//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldIntegrator.test.cc
//---------------------------------------------------------------------------//
#include "FieldTestBase.hh"

#include "field/FieldIntegrator.hh"
#include "field/FieldParamsPointers.hh"

#include "field/RungeKutta.hh"
#include "field/MagField.hh"
#include "field/FieldEquation.hh"
#include "field/base/OdeArray.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"

#include "celeritas_test.hh"

#ifdef CELERITAS_USE_CUDA
#    include "FieldIntegrator.test.hh"
#endif

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldIntegratorTest : public FieldTestBase
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FieldIntegratorTest, field_integrator_host)
{
    // Construct FieldIntegrator
    MagField        field({0, 0, test_params.field_value});
    FieldEquation   equation(field);
    RungeKutta      rk4(equation);
    FieldIntegrator integrator(field_params_view, rk4);

    // Initial state and the epected state after revolutions
    OdeArray y;
    y[0]        = test_params.radius;
    y[4]        = test_params.momentum;
    OdeArray yo = y;

    // The rhs of the equation and a temporary array
    OdeArray dydx;

    // Test parameters and the sub-step size
    real_type circumference = 2.0 * constants::pi * test_params.radius;

    test_params.nsteps = 100;
    double hstep       = circumference / test_params.nsteps;

    real_type total_error        = 0;
    real_type total_step_length  = 0;
    real_type total_curve_length = 0;
    real_type dist_chord, hnext;

    // Try the stepper by hstep for (num_revolutions * num_steps) times
    real_type delta = field_params_view.errcon;
    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        // test quick_advance
        OdeArray  y_quick = y;
        real_type dyerr   = 0;
        {
            equation(y_quick, dydx);
            dyerr = integrator.quick_advance(hstep, y_quick, dydx, dist_chord);
        }
        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(yo.get(), y.get(), delta);
        total_error += sqrt(dyerr);

        // test one_good_step
        OdeArray  y_good      = y;
        real_type step_length = 0;
        {
            equation(y_good, dydx);
            step_length = integrator.one_good_step(hstep, y_good, dydx, hnext);
        }

        // Check the step length and the state
        EXPECT_LT(fabs(step_length - hstep), delta);
        EXPECT_VEC_NEAR(yo.get(), y.get(), delta);
        total_step_length += step_length;

        // test accurate advance
        OdeArray  y_accurate   = y;
        real_type curve_length = 0;
        {
            equation(y_accurate, dydx);
            integrator.accurate_advance(hstep, y_accurate, curve_length, .001);
        }

        // Check the curve_length and the state
        EXPECT_LT(fabs(curve_length - hstep), delta);
        EXPECT_VEC_NEAR(yo.get(), y.get(), delta);
        total_curve_length += curve_length;

        // test find_next_chord
        OdeArray ystart = y;
        {
            integrator.advance_chord_limited(hstep, ystart);
        }
    }

    // Check the total error, step/curve length
    EXPECT_LT(total_error, delta * test_params.revolutions);
    EXPECT_LT(total_curve_length - circumference * test_params.revolutions,
              delta);
    EXPECT_LT(total_step_length - circumference * test_params.revolutions,
              delta);
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class FieldIntegratorDeviceTest : public FieldIntegratorTest
{
};

TEST_F(FieldIntegratorDeviceTest, field_integrator_device)
{
    // Run kernel
    test_params.nstates = 32;
    auto output         = integrator_test(field_params_view, test_params);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_LT(fabs(output.pos[i] - test_params.radius),
                  test_params.epsilon);
        EXPECT_LT(fabs(output.mom[i] - test_params.momentum),
                  test_params.epsilon);
        EXPECT_LT(output.err[i] - 2392.980, 1.0e-3);
    }
}

//---------------------------------------------------------------------------//
#endif
