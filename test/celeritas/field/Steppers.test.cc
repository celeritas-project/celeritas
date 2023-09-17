//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Steppers.test.cc
//---------------------------------------------------------------------------//

#include "Steppers.test.hh"

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/RungeKuttaStepper.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/field/ZHelixStepper.hh"

#include "FieldTestParams.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SteppersTest : public Test
{
  protected:
    void SetUp() override
    {
        /***
          Physical system of the test and parameters: the helix motion of
          an electron in a uniform magnetic field along the z-direction with
          initial velocity (v0), position (pos_0) and direction (dir_0).

          B     = {0.0, 0.0, 1.0 * units::tesla}
          v_0   = 0.999 * constants::c_light
          dir_0 = {0.0, 0.96, 0.28}

          gamma = 1.0/sqrt(1-ipow<2>(v0/constants::c_light))
          radius = constants::electron_mass*gamma *v0/(constants::e_electron*B)
          mass = constants::electron_mass*ipow<2>(constants::c_light)/MeV

          pos_0 = {radius, 0.0, 0.0}
          mom_0 = mass * sqrt(ipow<2>(gamma) - 1) * dir_0
        */

        param.field_value = 1.0 * units::tesla;  //! field value along z
                                                 //! [tesla]
        param.radius = 3.8085386036;  //! radius of curvature [cm]
        param.delta_z = 6.7003310629;  //! z-change/revolution [cm]
        param.momentum_y = 10.9610028286;  //! initial momentum_y [MeV/c]
        param.momentum_z = 3.1969591583;  //! initial momentum_z [MeV/c]
        param.nstates = 1;  //! number of states (tracks)
        param.nsteps = 100;  //! number of steps/revolution
        param.revolutions = 10;  //! number of revolutions
        param.epsilon = 1.0e-5;  //! tolerance error
    }

    template<class FieldT, template<class> class StepperT>
    void run_stepper(FieldT const& field)
    {
        // Construct a stepper for testing
        auto stepper = make_mag_field_stepper<StepperT>(
            field, units::ElementaryCharge{-1});
        // Test parameters and the sub-step size
        real_type hstep = 2.0 * constants::pi * param.radius / param.nsteps;

        for (unsigned int i : range(param.nstates))
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
                for ([[maybe_unused]] int j : range(param.nsteps))
                {
                    FieldStepperResult result = stepper(hstep, y);
                    y = result.end_state;

                    total_err2
                        += detail::rel_err_sq(result.err_state, hstep, y.mom);
                }
                real_type tol = std::sqrt(total_err2) / 0.001;
                // Check the state after each revolution and the total error
                EXPECT_VEC_CLOSE(expected_y.pos, y.pos, tol, tol);
                EXPECT_VEC_CLOSE(expected_y.mom, y.mom, tol, tol);
                EXPECT_LT(total_err2, param.epsilon);
            }
        }
    }

    void check_result(StepperTestOutput const& output)
    {
        // Check gpu stepper results
        real_type zstep = param.delta_z * param.revolutions;
        for (auto i : range(output.pos_x.size()))
        {
            real_type error = std::sqrt(output.error[i]);
            EXPECT_SOFT_NEAR(output.pos_x[i], param.radius, error);
            EXPECT_SOFT_NEAR(output.pos_z[i], zstep + i * 1.0e-6, error);
            EXPECT_SOFT_NEAR(output.mom_y[i], param.momentum_y, error);
            EXPECT_SOFT_NEAR(output.mom_z[i], param.momentum_z, error);
            EXPECT_LT(output.error[i], param.epsilon);
        }
    }

    // Test parameters
    FieldTestParams param;
};

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//
TEST_F(SteppersTest, host_helix)
{
    // Construct a uniform magnetic field along Z axis
    UniformZField field(param.field_value);

    // Test the analytical ZHelix stepper
    this->run_stepper<UniformZField, ZHelixStepper>(field);
}

//---------------------------------------------------------------------------//
TEST_F(SteppersTest, host_classical_rk4)
{
    // Construct a uniform magnetic field
    UniformField field({0, 0, param.field_value});

    // Test the classical 4th order Runge-Kutta stepper
    this->run_stepper<UniformField, RungeKuttaStepper>(field);
}

//---------------------------------------------------------------------------//
TEST_F(SteppersTest, host_dormand_prince_547)
{
    // Construct a uniform magnetic field
    UniformField field({0, 0, param.field_value});

    // Test the Dormand-Prince 547(M) stepper
    this->run_stepper<UniformField, DormandPrinceStepper>(field);
}  //---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
