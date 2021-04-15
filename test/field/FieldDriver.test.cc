//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.test.cc
//---------------------------------------------------------------------------//

#include "field/FieldDriver.hh"
#include "field/FieldParamsPointers.hh"
#include "field/FieldInterface.hh"

#include "field/RungeKuttaStepper.hh"
#include "field/MagField.hh"
#include "field/MagFieldEquation.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"

#include "celeritas_test.hh"

#ifdef CELERITAS_USE_CUDA
#    include "FieldDriver.test.hh"
#endif

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldDriverTest : public Test
{
  protected:
    void SetUp() override
    {
        // Set values of FieldParamsPointers;
        field_params.minimum_step          = 1.0e-5 * units::millimeter;
        field_params.delta_chord           = 0.25 * units::millimeter;
        field_params.delta_intersection    = 1.0e-4 * units::millimeter;
        field_params.epsilon_step          = 1.0e-5;
        field_params.epsilon_rel_max       = 1.0e-3;
        field_params.errcon                = 1.0e-4;
        field_params.pgrow                 = -0.20;
        field_params.pshrink               = -0.25;
        field_params.safety                = 0.9;
        field_params.max_stepping_increase = 5;
        field_params.max_stepping_decrease = 0.1;
        field_params.max_nsteps            = 100;

        // Input parameters of an electron in a uniform magnetic field
        test_params.nstates     = 128 * 512;
        test_params.nsteps      = 100;
        test_params.revolutions = 10;
        test_params.field_value = 1.0 * units::tesla;
        test_params.radius      = 3.8085386036 * units::centimeter;
        test_params.delta_z     = 6.7003310629 * units::centimeter;
        test_params.energy      = 10.9181415106;
        test_params.momentum_y  = 10.9610028286;
        test_params.momentum_z  = 3.1969591583;
        test_params.epsilon     = 1.0e-5;
    }

  protected:
    // Field parameters
    FieldParamsPointers field_params;

    // Test parameters
    FieldTestParams test_params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FieldDriverTest, field_driver_host)
{
    // Construct FieldDriver
    MagField         field({0, 0, test_params.field_value});
    MagFieldEquation equation(field, units::ElementaryCharge{-1});
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep         = circumference / test_params.nsteps;

    // Only test every 128 states to reduce debug runtime
    for (unsigned int i : celeritas::range(test_params.nstates).step(128u))
    {
        // Initial state and the epected state after revolutions
        OdeState y;
        y.pos = {test_params.radius, 0, i * 1.0e-6};
        y.mom = {0, test_params.momentum_y, test_params.momentum_z};

        OdeState y_expected = y;

        real_type total_step_length{0};

        // Try the stepper by hstep for (num_revolutions * num_steps) times
        real_type delta = field_params.errcon;
        for (int nr = 0; nr < test_params.revolutions; ++nr)
        {
            y_expected.pos = {test_params.radius,
                              0,
                              (nr + 1) * test_params.delta_z + i * 1.0e-6};

            // Travel hstep for num_steps times in the field
            for (CELER_MAYBE_UNUSED int j : range(test_params.nsteps))
            {
                total_step_length += driver(hstep, &y);
            }

            // Check the total error and the state (position, momentum)
            EXPECT_VEC_NEAR(y_expected.pos, y.pos, delta);
        }

        // Check the total error, step/curve length
        EXPECT_SOFT_NEAR(
            total_step_length, circumference * test_params.revolutions, delta);
    }
}

TEST_F(FieldDriverTest, accurate_advance_host)
{
    // Construct FieldDriver
    MagField         field({0, 0, test_params.field_value});
    MagFieldEquation equation(field, units::ElementaryCharge{-1});
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep         = circumference / test_params.nsteps;

    // Only test every 128 states to reduce debug runtime
    for (unsigned int i : celeritas::range(test_params.nstates).step(128u))
    {
        // Initial state and the epected state after revolutions
        OdeState y;
        y.pos = {test_params.radius, 0, i * 1.0e-6};
        y.mom = {0, test_params.momentum_y, test_params.momentum_z};

        OdeState y_expected = y;

        // Try the stepper by hstep for (num_revolutions * num_steps) times
        real_type total_curved_length{0};
        real_type delta = field_params.errcon;

        for (int nr = 0; nr < test_params.revolutions; ++nr)
        {
            // test one_good_step
            OdeState y_accurate = y;

            // Travel hstep for num_steps times in the field
            for (CELER_MAYBE_UNUSED int j : range(test_params.nsteps))
            {
                total_curved_length
                    += driver.accurate_advance(hstep, &y_accurate, .001);
            }
            // Check the total error and the state (position, momentum)
            EXPECT_VEC_NEAR(y_expected.pos, y.pos, delta);
        }

        // Check the total error, step/curve length
        EXPECT_LT(total_curved_length - circumference * test_params.revolutions,
                  delta);
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class FieldDriverDeviceTest : public FieldDriverTest
{
};

TEST_F(FieldDriverDeviceTest, TEST_IF_CELERITAS_CUDA(field_driver_device))
{
    // Run kernel
    auto output = driver_test(field_params, test_params);

    // Check stepper results
    real_type zstep = test_params.delta_z * test_params.revolutions;
    real_type delta = field_params.errcon;

    real_type circumference = 2 * constants::pi * test_params.radius;

    for (auto i : range(test_params.nstates))
    {
        EXPECT_SOFT_NEAR(output.pos_x[i], test_params.radius, delta);
        EXPECT_SOFT_NEAR(output.pos_z[i], zstep + i * 1.0e-6, delta);
        EXPECT_SOFT_NEAR(output.mom_y[i], test_params.momentum_y, delta);
        EXPECT_SOFT_NEAR(output.mom_z[i], test_params.momentum_z, delta);
        EXPECT_SOFT_NEAR(
            output.error[i], circumference * test_params.revolutions, delta);
    }
}

TEST_F(FieldDriverDeviceTest, TEST_IF_CELERITAS_CUDA(accurate_advance_device))
{
    // Run kernel
    auto output = accurate_advance_test(field_params, test_params);

    // Check stepper results
    real_type zstep = test_params.delta_z;
    real_type delta = field_params.errcon;

    real_type circumference = 2 * constants::pi * test_params.radius;

    for (auto i : range(test_params.nstates))
    {
        EXPECT_SOFT_NEAR(output.pos_x[i], test_params.radius, delta);
        EXPECT_SOFT_NEAR(output.pos_z[i], zstep + i * 1.0e-6, delta);
        EXPECT_SOFT_NEAR(output.mom_y[i], test_params.momentum_y, delta);
        EXPECT_SOFT_NEAR(output.mom_z[i], test_params.momentum_z, delta);
        EXPECT_SOFT_NEAR(
            output.length[i], circumference * test_params.revolutions, delta);
    }
}

//---------------------------------------------------------------------------//
#endif
