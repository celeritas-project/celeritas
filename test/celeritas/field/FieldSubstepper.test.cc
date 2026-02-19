//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldSubstepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/FieldSubstepper.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/Types.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/field/ZHelixIntegrator.hh"
#include "celeritas/field/detail/FieldUtils.hh"

#include "DiagnosticIntegrator.hh"
#include "FieldTestParams.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using namespace celeritas::units::literals;

constexpr real_type sqrt_two{constants::sqrt_two};
using units::ElementaryCharge;
using units::MevEnergy;
using units::MevMass;
using units::MevMomentum;

template<class E>
using DiagnosticDPIntegrator = DiagnosticIntegrator<DormandPrinceIntegrator<E>>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldSubstepperTest : public Test
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
        return this->calc_momentum(energy).value() * dir;
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

class RevolutionFieldSubstepperTest : public FieldSubstepperTest
{
  protected:
    void SetUp() override
    {
        // Input parameters of an electron in a uniform magnetic field
        test_params.nsteps = 100;
        test_params.revolutions = 10;
        test_params.radius = 3.8085386036_cm;
        test_params.delta_z = 6.7003310629_cm;
        test_params.epsilon = 1.0e-5;
    }

  protected:
    // Field parameters
    FieldDriverOptions driver_options;

    // Test parameters
    FieldTestParams test_params;
};

//---------------------------------------------------------------------------//

template<template<class EquationT> class IntegratorT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_substepper(FieldT&& field,
                          FieldDriverOptions const& options,
                          units::ElementaryCharge charge)
{
    using Substepper_t = FieldSubstepper<IntegratorT<MagFieldEquation<FieldT>>>;
    return Substepper_t{options,
                        make_mag_field_integrator<IntegratorT>(
                            ::celeritas::forward<FieldT>(field), charge)};
}

//! Z oriented field of 2**(y / scale)
struct ExpZField
{
    real_type strength{1_T};
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

TEST_F(FieldSubstepperTest, types)
{
    FieldDriverOptions driver_options;
    auto advance = make_mag_field_substepper<DormandPrinceIntegrator>(
        UniformZField(1), driver_options, electron_charge());

    // Make sure object is holding things by value
    EXPECT_TRUE((std::is_same<FieldSubstepper<DormandPrinceIntegrator<
                                  MagFieldEquation<UniformZField>>>,
                              decltype(advance)>::value));
    // Size: field vector, q / c, reference to options

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(3 * sizeof(real_type) + sizeof(FieldDriverOptions*),
                  sizeof(advance));
    }
}

// Field strength changes quickly with z, so different chord steps require
// different substeps
TEST_F(FieldSubstepperTest, unpleasant_field)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = 32;

    real_type field_strength = 1.0_T;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    // Vary by a factor of 1024 over the radius of curvature
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        ExpZField{field_strength, radius / 10}, units::ElementaryCharge{-1});
    FieldSubstepper advance{driver_options, integrate};

    OdeState state;
    state.pos = {radius, 0, 0};
    state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

    real_type distance{0};
    for (auto i : range(1, 6))
    {
        auto result = advance(from_cm(i), state);
        distance += result.length;
        state = result.state;
    }
    EXPECT_EQ(20, integrate.count());
    EXPECT_SOFT_EQ(2.0197620480043263, distance);
}

// As the track moves along +z near 0, the field strength oscillates horribly,
// so the "one good step" convergence requires more than one iteration (which
// doesn't happen for any of the other more well-behaved fields).
TEST_F(FieldSubstepperTest, horrible_field)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = 32;

    real_type field_strength = 1.0_T;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        HorribleZField{field_strength, radius / 10},
        units::ElementaryCharge{-1});
    FieldSubstepper advance{driver_options, integrate};

    OdeState state;
    state.pos = {radius, 0, -radius / 5};
    state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

    real_type accum{0};
    for (int i = 1; i < 5; ++i)
    {
        auto result = advance(from_cm(0.05), state);
        accum += result.length;
        state = result.state;
    }
    EXPECT_EQ(9, integrate.count());
    EXPECT_SOFT_EQ(0.2, accum);
    EXPECT_SOFT_NEAR(0,
                     distance(Real3({0.49120878051539413,
                                     0.14017717257531165,
                                     0.04668993728754612}),
                              state.pos),
                     coarse_eps * 10)
        << state.pos;
}

/*!
 * Demonstrate the misbehavior of the chord finder for tightly circling
 * particles.
 */
TEST_F(FieldSubstepperTest, pathological_chord)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = std::numeric_limits<short int>::max();

    real_type field_strength = 1.0_T;
    MevEnergy e{1.0};
    real_type radius = this->calc_curvature(e, field_strength);

    OdeState state;
    state.pos = {radius, 0, 0};
    state.mom = this->calc_momentum(
        e, {0, std::sqrt(1 - ipow<2>(real_type{0.2})), 0.2});

    DiagnosticIntegrator integrate{ZHelixIntegrator{MagFieldEquation{
        UniformZField{field_strength}, units::ElementaryCharge{-1}}}};
    FieldSubstepper advance{driver_options, integrate};

    std::vector<unsigned int> counts;
    std::vector<real_type> lengths;

    for (auto rev : {0.01, 1.0, 2.0, 4.0, 8.0})
    {
        integrate.reset_count();
        auto end = advance(rev * 2 * constants::pi * radius, state);
        counts.push_back(integrate.count());
        lengths.push_back(end.length);
    }

    static unsigned int const expected_counts[] = {1u, 6u, 4u, 4u, 4u};
    static double const expected_lengths[] = {0.029802281646312,
                                              0.30937398137671,
                                              0.30936881116327,
                                              0.30936881114832,
                                              0.30936881114832};
    EXPECT_VEC_EQ(expected_counts, counts);
    EXPECT_VEC_SOFT_EQ(expected_lengths, lengths);
}

TEST_F(FieldSubstepperTest, step_counts)
{
    FieldDriverOptions driver_options;
    driver_options.max_nsteps = std::numeric_limits<short int>::max();

    real_type field_strength = 1.0_T;
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        UniformZField{field_strength}, units::ElementaryCharge{-1});

    std::vector<real_type> radii;
    std::vector<unsigned int> counts;
    std::vector<real_type> lengths;

    // Test the number of field equation evaluations that have to be done to
    // travel a step length of 1e-4 cm and 10 cm, for electrons from 0.1 eV to
    // 10 TeV.
    for (int loge : range(-7, 7).step(2))
    {
        MevEnergy e{std::pow(real_type{10}, static_cast<real_type>(loge))};
        real_type radius = this->calc_curvature(e, field_strength);
        radii.push_back(radius);

        OdeState state;
        state.pos = {radius, 0, 0};
        state.mom = this->calc_momentum(e, {0, sqrt_two / 2, sqrt_two / 2});

        FieldSubstepper advance{driver_options, integrate};
        for (int log_len : range(-4, 3).step(2))
        {
            real_type step_len = std::pow(10.0, log_len);
            integrate.reset_count();
            auto end = advance(step_len * units::centimeter, state);

            counts.push_back(integrate.count());
            lengths.push_back(end.length);
        }
    }

    // clang-format off
    static double const expected_radii[] = {0.00010663611598835,
        0.0010663663247419, 0.010668826843187, 0.11173141982667,
        3.5019461121752, 333.73450257138, 33356.579970281};
    static unsigned int const expected_counts[] = {1u, 93u, 779u, 777u, 1u,
        12u, 90u, 87u, 1u, 1u, 29u, 25u, 1u, 1u, 7u, 5u, 1u, 1u, 2u, 3u, 1u,
        1u, 1u, 5u, 1u, 1u, 1u, 2u};
    static double const expected_lengths[] = {0.0001, 0.01, 0.077563521220272,
        0.077562363386602, 0.0001, 0.01, 0.076209386999884, 0.076209671160348,
        0.0001, 0.01, 0.063064075311856, 0.063065174124004, 0.0001, 0.01,
        0.17398853544975, 0.17398853544975, 0.0001, 0.01, 0.99607291767799,
        0.99607023941998, 0.0001, 0.01, 1, 9.7158185571513, 0.0001, 0.01, 1,
        97.132215683182};
    // clang-format on

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_SOFT_EQ(expected_radii, radii);
        EXPECT_VEC_EQ(expected_counts, counts);
        EXPECT_VEC_SOFT_EQ(expected_lengths, lengths);
    }
}

//---------------------------------------------------------------------------//

TEST_F(RevolutionFieldSubstepperTest, advance)
{
    auto advance = make_mag_field_substepper<DormandPrinceIntegrator>(
        UniformZField{1.0_T}, driver_options, electron_charge());

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep = circumference / test_params.nsteps;

    // Initial state and the expected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
    y.mom = this->calc_momentum(MevEnergy{10.9181415106}, {0, 0.96, 0.28});

    OdeState y_expected = y;

    real_type total_step_length{0};

    // Try the integrate by hstep for (num_revolutions * num_steps) times
    real_type eps = 1.0e-4;
    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        y_expected.pos
            = {test_params.radius, 0, (nr + 1) * test_params.delta_z};

        // Travel hstep for num_steps times in the field
        for ([[maybe_unused]] int j : range(test_params.nsteps))
        {
            auto end = advance(hstep, y);
            total_step_length += end.length;
            y = end.state;
        }

        // Check the total error and the state (position, momentum)
        EXPECT_VEC_NEAR(y_expected.pos, y.pos, (SoftEqual{eps, eps}));
    }

    // Check the total error, step/curve length
    EXPECT_SOFT_NEAR(
        total_step_length, circumference * test_params.revolutions, eps);
}

TEST_F(RevolutionFieldSubstepperTest, accurate_advance)
{
    auto advance = make_mag_field_substepper<DormandPrinceIntegrator>(
        UniformZField{1.0_T}, driver_options, electron_charge());

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep = circumference / test_params.nsteps;

    // Initial state and the expected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, 0};
    y.mom = this->calc_momentum(MevEnergy{10.9181415106}, {0, 0.96, 0.28});

    OdeState y_expected = y;

    // Try the integrate by hstep for (num_revolutions * num_steps) times
    real_type total_curved_length{0};
    real_type eps = std::sqrt(this->coarse_eps);

    for (int nr = 0; nr < test_params.revolutions; ++nr)
    {
        // test one_good_step
        OdeState y_accurate = y;

        // Travel hstep for num_steps times in the field
        for ([[maybe_unused]] int j : range(test_params.nsteps))
        {
            auto end = advance.accurate_advance(hstep, y_accurate, .001);

            total_curved_length += end.length;
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
