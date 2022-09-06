//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/field/FieldDriver.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/Types.hh"
#include "celeritas/field/UniformField.hh"

#include "FieldTestParams.hh"
#include "celeritas_test.hh"

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
        // Input parameters of an electron in a uniform magnetic field
        test_params.nstates     = 1;
        test_params.nsteps      = 100;
        test_params.revolutions = 10;
        test_params.field_value = 1.0 * units::tesla;
        test_params.radius      = 3.8085386036 * units::centimeter;
        test_params.delta_z     = 6.7003310629 * units::centimeter;
        test_params.energy      = 10.9181415106; // MeV
        test_params.momentum_y  = 10.9610028286; // MeV/c
        test_params.momentum_z  = 3.1969591583;  // MeV/c
        test_params.epsilon     = 1.0e-5;
    }

  protected:
    // Field parameters
    FieldDriverOptions field_params;

    // Test parameters
    FieldTestParams test_params;
};

//---------------------------------------------------------------------------//

template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_driver(FieldT&&                             field,
                      const celeritas::FieldDriverOptions& options,
                      celeritas::units::ElementaryCharge   charge)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;
    using Stepper_t  = StepperT<Equation_t>;
    using Driver_t   = celeritas::FieldDriver<Stepper_t>;
    return Driver_t{
        options,
        Stepper_t{Equation_t{::celeritas::forward<FieldT>(field), charge}}};
}
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FieldDriverTest, field_driver_host)
{
    // Construct FieldDriver
    auto driver = make_mag_field_driver<DormandPrinceStepper>(
        UniformField({0, 0, test_params.field_value}),
        field_params,
        units::ElementaryCharge{-1});

    // Make sure object is holding things by value
    EXPECT_TRUE(
        (std::is_same<
            FieldDriver<DormandPrinceStepper<MagFieldEquation<UniformField>>>,
            decltype(driver)>::value));
    // Size: field vector, q / c, reference to options
    EXPECT_EQ(sizeof(Real3) + sizeof(real_type) + sizeof(FieldDriverOptions*),
              sizeof(driver));

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep         = circumference / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
    y.mom = {0, test_params.momentum_y, test_params.momentum_z};

    OdeState y_expected = y;

    real_type total_step_length{0};

    // Try the stepper by hstep for (num_revolutions * num_steps) times
    real_type delta = field_params.errcon;
    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        y_expected.pos
            = {test_params.radius, 0, (nr + 1) * test_params.delta_z};

        // Travel hstep for num_steps times in the field
        for (CELER_MAYBE_UNUSED int j : range(test_params.nsteps))
        {
            auto end = driver.advance(hstep, y);
            total_step_length += end.step;
            y = end.state;
        }

        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(y_expected.pos, y.pos, delta);
    }

    // Check the total error, step/curve length
    EXPECT_SOFT_NEAR(
        total_step_length, circumference * test_params.revolutions, delta);
}

TEST_F(FieldDriverTest, accurate_advance_host)
{
    auto driver = make_mag_field_driver<DormandPrinceStepper>(
        UniformField({0, 0, test_params.field_value}),
        field_params,
        units::ElementaryCharge{-1});

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep         = circumference / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
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
            auto end = driver.accurate_advance(hstep, y_accurate, .001);

            total_curved_length += end.step;
            y_accurate = end.state;
        }
        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(y_expected.pos, y.pos, delta);
    }

    // Check the total error, step/curve length
    EXPECT_LT(total_curved_length - circumference * test_params.revolutions,
              delta);
}