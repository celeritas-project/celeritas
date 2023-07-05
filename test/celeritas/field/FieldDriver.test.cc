//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/field/FieldDriver.hh"

#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/Types.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/field/ZHelixStepper.hh"
#include "celeritas/field/detail/FieldUtils.hh"

#include "DiagnosticStepper.hh"
#include "FieldTestParams.hh"
#include "celeritas_test.hh"

using celeritas::constants::sqrt_two;
using celeritas::units::ElementaryCharge;
using celeritas::units::MevEnergy;
using celeritas::units::MevMass;
using celeritas::units::MevMomentum;

template<class E>
using DiagnosticDPStepper
    = celeritas::test::DiagnosticStepper<celeritas::DormandPrinceStepper<E>>;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldDriverTest : public Test
{
  public:
    static constexpr MevMass electron_mass() { return MevMass{0.5109989461}; }
    static constexpr ElementaryCharge electron_charge()
    {
        return ElementaryCharge{-1};
    }

    // Calculate momentum assuming an electron
    MevMomentum calc_momentum(MevEnergy energy) const
    {
        real_type m = electron_mass().value();
        real_type e = energy.value();
        return MevMomentum{std::sqrt(e * e + 2 * m * e)};
    }

    // Get the momentum in units of MevMomentum
    Real3 calc_momentum(MevEnergy energy, Real3 const& dir)
    {
        CELER_EXPECT(is_soft_unit_vector(dir));
        return detail::ax(this->calc_momentum(energy).value(), dir);
    }

    // Calculate momentum assuming an electron
    real_type calc_curvature(MevEnergy energy, real_type field_strength) const
    {
        CELER_EXPECT(field_strength > 0);
        return native_value_from(calc_momentum(energy))
               / (std::fabs(native_value_from(electron_charge()))
                  * field_strength);
    }
};

class RevolutionFieldDriverTest : public FieldDriverTest
{
  protected:
    void SetUp() override
    {
        // Input parameters of an electron in a uniform magnetic field
        test_params.nsteps = 100;
        test_params.revolutions = 10;
        test_params.radius = 3.8085386036 * units::centimeter;
        test_params.delta_z = 6.7003310629 * units::centimeter;
        test_params.epsilon = 1.0e-5;
    }

  protected:
    // Field parameters
    FieldDriverOptions driver_options;

    // Test parameters
    FieldTestParams test_params;
};

//---------------------------------------------------------------------------//

template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_driver(FieldT&& field,
                      FieldDriverOptions const& options,
                      units::ElementaryCharge charge)
{
    using Driver_t = FieldDriver<StepperT<MagFieldEquation<FieldT>>>;
    return Driver_t{options,
                    make_mag_field_stepper<StepperT>(
                        ::celeritas::forward<FieldT>(field), charge)};
}

//! Z oriented field of 2**(y / scale)
struct ExpZField
{
    real_type strength{1 * units::tesla};
    real_type scale{1};

    Real3 operator()(Real3 const& pos) const
    {
        return {0, 0, this->strength * std::exp2(pos[1] / scale)};
    }
};

//! sin(1/z), scaled and with multiplicative constant
struct HorribleZField
{
    real_type strength{1};
    real_type scale{1};

    Real3 operator()(Real3 const& pos) const
    {
        return {0, 0, this->strength * std::sin(this->scale / pos[2])};
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FieldDriverTest, types)
{
    FieldDriverOptions driver_options;
    auto driver = make_mag_field_driver<DormandPrinceStepper>(
        UniformZField(1), driver_options, electron_charge());

    // Make sure object is holding things by value
    EXPECT_TRUE(
        (std::is_same<
            FieldDriver<DormandPrinceStepper<MagFieldEquation<UniformZField>>>,
            decltype(driver)>::value));
    // Size: field vector, q / c, reference to options
    EXPECT_EQ(
        sizeof(real_type) + sizeof(real_type) + sizeof(FieldDriverOptions*),
        sizeof(driver));
}

// Field strength changes quickly with z, so different chord steps require
// different substeps
TEST_F(FieldDriverTest, unpleasant_field)
{
    constexpr auto cm = units::centimeter;

    FieldDriverOptions driver_options;
    driver_options.max_nsteps = 32;

    real_type field_strength = 1.0 * units::tesla;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    // Vary by a factor of 1024 over the radius of curvature
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        ExpZField{field_strength, radius / 10}, units::ElementaryCharge{-1});
    FieldDriver driver{driver_options, stepper};

    OdeState state;
    state.pos = {radius, 0, 0};
    state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

    real_type distance{0};
    for (auto i : range(1, 6))
    {
        auto result = driver.advance(i * cm, state);
        distance += result.step;
        state = result.state;
    }
    EXPECT_EQ(28, stepper.count());
    EXPECT_SOFT_EQ(9.7011773019513114, distance);
}

// As the track moves along +z near 0, the field strength oscillates horribly,
// so the "one good step" convergence requires more than one iteration (which
// doesn't happen for any of the other more well-behaved fields).
TEST_F(FieldDriverTest, horrible_field)
{
    constexpr auto cm = units::centimeter;

    FieldDriverOptions driver_options;
    driver_options.max_nsteps = 32;

    real_type field_strength = 1.0 * units::tesla;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        HorribleZField{field_strength, radius / 10},
        units::ElementaryCharge{-1});
    FieldDriver driver{driver_options, stepper};

    OdeState state;
    state.pos = {radius, 0, -radius / 5};
    state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

    real_type accum{0};
    for (int i = 1; i < 5; ++i)
    {
        auto result = driver.advance(0.05 * cm, state);
        accum += result.step;
        state = result.state;
    }
    EXPECT_EQ(9, stepper.count());
    EXPECT_SOFT_EQ(0.2, accum);
    EXPECT_SOFT_NEAR(0,
                     distance(Real3({0.49120878051539413,
                                     0.14017717257531165,
                                     0.04668993728754612}),
                              state.pos),
                     1e-5)
        << state.pos;
}

TEST_F(FieldDriverTest, pathological_chord)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = std::numeric_limits<short int>::max();

    real_type field_strength = 1.0 * units::tesla;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    OdeState state;
    state.pos = {radius, 0, 0};
    state.mom = this->calc_momentum(e, {0, std::sqrt(1 - ipow<2>(0.2)), 0.2});

    DiagnosticStepper stepper{ZHelixStepper{MagFieldEquation{
        UniformZField{field_strength}, units::ElementaryCharge{-1}}}};
    FieldDriver driver{driver_options, stepper};

    std::vector<unsigned int> counts;
    std::vector<real_type> lengths;

    real_type const eps = 0;
    for (auto rev : {0.01, 1.0, 2.0, 4.0, 8.0})
    {
        stepper.reset_count();
        auto end
            = driver.advance((rev + eps) * 2 * constants::pi * radius, state);
        counts.push_back(stepper.count());
        lengths.push_back(end.step);
    }

    static unsigned int const expected_counts[] = {1u, 6u, 5u, 6u, 7u};
    static double const expected_lengths[] = {0.029802281646312,
                                              0.30936865386382,
                                              0.30936877899171,
                                              0.30936877899171,
                                              0.3093686200038};
    EXPECT_VEC_EQ(expected_counts, counts);
    EXPECT_VEC_SOFT_EQ(expected_lengths, lengths);
}

TEST_F(FieldDriverTest, step_counts)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = std::numeric_limits<short int>::max();

    real_type field_strength = 1.0 * units::tesla;
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        UniformZField{field_strength}, units::ElementaryCharge{-1});
    FieldDriver driver{driver_options, stepper};

    std::vector<real_type> radii;
    std::vector<unsigned int> counts;
    std::vector<real_type> lengths;

    // Test the number of field equation evaluations that have to be done to
    // travel a step length of 1e-4 cm and 10 cm, for electrons from 0.1 eV to
    // 10 TeV.
    for (int loge : range(-7, 7).step(2))
    {
        MevEnergy e{std::pow(10.0, loge)};
        real_type radius = this->calc_curvature(e, field_strength);
        radii.push_back(radius);

        OdeState state;
        state.pos = {radius, 0, 0};
        state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

        for (int log_len : range(-4, 3).step(2))
        {
            real_type step_len = std::pow(10.0, log_len);
            stepper.reset_count();
            auto end = driver.advance(step_len * units::centimeter, state);

            counts.push_back(stepper.count());
            lengths.push_back(end.step);
        }
    }

    // clang-format off
    static double const expected_radii[] = {0.00010663611598835,
        0.0010663663247419, 0.010668826843187, 0.11173141982667,
        3.5019461121752, 333.73450257138, 33356.579970281};
    static unsigned int const expected_counts[] = {1u, 8u, 13u, 18u, 1u, 6u,
        12u, 15u, 1u, 1u, 8u, 13u, 1u, 1u, 8u, 11u, 1u, 1u, 2u, 8u, 1u, 1u, 1u,
        5u, 1u, 1u, 1u, 2u};
    static double const expected_lengths[] = {0.0001, 0.00022174891924897,
        0.00027668262492734, 0.00034567495046535, 0.0001, 0.0018937958946702,
        0.0024805903384976, 0.0030891846584462, 0.0001, 0.01,
        0.022178208864978, 0.027672274443162, 0.0001, 0.01, 0.17883331913339,
        0.1734492975358, 0.0001, 0.01, 0.99607291767799, 0.99606877280876,
        0.0001, 0.01, 1, 9.7158185571513, 0.0001, 0.01, 1, 97.132215683182};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_radii, radii);
    EXPECT_VEC_EQ(expected_counts, counts);
    EXPECT_VEC_SOFT_EQ(expected_lengths, lengths);
}

//---------------------------------------------------------------------------//

TEST_F(RevolutionFieldDriverTest, advance)
{
    auto driver = make_mag_field_driver<DormandPrinceStepper>(
        UniformZField{1.0 * units::tesla}, driver_options, electron_charge());

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep = circumference / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
    y.mom = this->calc_momentum(MevEnergy{10.9181415106}, {0, 0.96, 0.28});

    OdeState y_expected = y;

    real_type total_step_length{0};

    // Try the stepper by hstep for (num_revolutions * num_steps) times
    real_type eps = 1.0e-4;
    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        y_expected.pos
            = {test_params.radius, 0, (nr + 1) * test_params.delta_z};

        // Travel hstep for num_steps times in the field
        for ([[maybe_unused]] int j : range(test_params.nsteps))
        {
            auto end = driver.advance(hstep, y);
            total_step_length += end.step;
            y = end.state;
        }

        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(y_expected.pos, y.pos, eps);
    }

    // Check the total error, step/curve length
    EXPECT_SOFT_NEAR(
        total_step_length, circumference * test_params.revolutions, eps);
}

TEST_F(RevolutionFieldDriverTest, accurate_advance)
{
    auto driver = make_mag_field_driver<DormandPrinceStepper>(
        UniformZField{1.0 * units::tesla}, driver_options, electron_charge());

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep = circumference / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
    y.mom = this->calc_momentum(MevEnergy{10.9181415106}, {0, 0.96, 0.28});

    OdeState y_expected = y;

    // Try the stepper by hstep for (num_revolutions * num_steps) times
    real_type total_curved_length{0};
    real_type eps = 1.0e-4;

    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        // test one_good_step
        OdeState y_accurate = y;

        // Travel hstep for num_steps times in the field
        for ([[maybe_unused]] int j : range(test_params.nsteps))
        {
            auto end = driver.accurate_advance(hstep, y_accurate, .001);

            total_curved_length += end.step;
            y_accurate = end.state;
        }
        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(y_expected.pos, y.pos, eps);
    }

    // Check the total error, step/curve length
    EXPECT_LT(total_curved_length - circumference * test_params.revolutions,
              eps);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
