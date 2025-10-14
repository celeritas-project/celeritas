//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Fresnel.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/optical/surface/model/DielectricInteractor.hh"
#include "celeritas/optical/surface/model/FresnelCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
/*!
 * In the \c CoordinateAxes definition, the transverse-electric mode is the
 * polarization component along \f(\hat{p}\f), and the transverse-magnetic mode
 * is the remaining component.
 */
struct LinearPolarization
{
    real_type t_e;
    real_type t_m;
};

static LinearPolarization const TE{1, 0};
static LinearPolarization const TM{0, 1};

//---------------------------------------------------------------------------//
/*!
 * Coordinate frame for a surface normal with an incident photon.
 *
 * The \f(\hat{n}\f) axis is the surface normal. If the incident photon
 * direction is antiparallel to the normal, then \f(\hat{p}\f) is the photon
 * polarization. Otherwise, \f(\hat{s}\f) is the orthogonal component of the
 * direction from the normal. In both cases, the remaining vector is defined
 * through the remaining cross product.
 */
struct CoordinateAxes
{
    real_type rel_r_index;

    Real3 n_hat;  //!< surface normal
    Real3 s_hat;  //!< direction in plane (along photon direction)
    Real3 p_hat;  //!< out-of-plane direction

    explicit operator bool() const
    {
        return rel_r_index > 0 && soft_zero(dot_product(n_hat, s_hat))
               && soft_zero(dot_product(n_hat, p_hat))
               && soft_zero(dot_product(s_hat, p_hat));
    }

    Real3 make_direction(real_type inc_angle) const
    {
        return std::sin(inc_angle) * s_hat - std::cos(inc_angle) * n_hat;
    }

    Real3
    make_polarization(real_type inc_angle, LinearPolarization const& pol) const
    {
        return make_unit_vector(pol.t_e * p_hat
                                + pol.t_m
                                      * (std::cos(inc_angle) * s_hat
                                         + std::sin(inc_angle) * n_hat));
    }

    real_type
    calc_reflectivity(real_type angle, LinearPolarization const& pol) const
    {
        return FresnelCalculator{this->make_direction(angle),
                                 this->make_polarization(angle, pol),
                                 n_hat,
                                 rel_r_index}
            .calc_reflectivity();
    }

    SurfaceInteraction
    calc_refraction(real_type angle, LinearPolarization const& pol) const
    {
        return FresnelCalculator{this->make_direction(angle),
                                 this->make_polarization(angle, pol),
                                 n_hat,
                                 rel_r_index}
            .refracted_interaction();
    }
};

struct ScatteringResult
{
    std::vector<real_type> cos_theta;
    std::vector<real_type> s_component;
    std::vector<real_type> p_component;
};

ScatteringResult scan_refraction(CoordinateAxes const& axes,
                                 LinearPolarization const& pol,
                                 std::vector<real_type> const& angles)
{
    ScatteringResult result;

    for (auto angle : angles)
    {
        auto refract = axes.calc_refraction(angle, pol);

        EXPECT_EQ(SurfaceInteraction::Action::refracted, refract.action);
        EXPECT_SOFT_EQ(0, dot_product(refract.direction, axes.p_hat));

        real_type cos_theta = clamp(-dot_product(refract.direction, axes.n_hat),
                                    real_type{0},
                                    real_type{1});
        real_type theta = std::acos(cos_theta);

        result.cos_theta.push_back(cos_theta);
        result.s_component.push_back(dot_product(
            refract.polarization, axes.make_polarization(theta, TE)));
        result.p_component.push_back(dot_product(
            refract.polarization, axes.make_polarization(theta, TM)));
    }

    return result;
}

//---------------------------------------------------------------------------//
// TEST CHASIS
//---------------------------------------------------------------------------//

class FresnelTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}

    void check_special_reflectivity_cases(CoordinateAxes const& axes)
    {
        // Reflectivities equal at normal incidence
        EXPECT_SOFT_EQ(axes.calc_reflectivity(0, TE),
                       axes.calc_reflectivity(0, TM));

        // Brewster angle has zero TM reflection
        real_type brewster_angle = std::atan(axes.rel_r_index);
        EXPECT_SOFT_EQ(0, axes.calc_reflectivity(brewster_angle, TM));
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Scan reflectivities for external reflection
TEST_F(FresnelTest, external_reflectivity)
{
    // External reflectivity has relative index > 1
    CoordinateAxes axes{13.0 / 7.0,
                        make_unit_vector(Real3{-2, 1, -1}),
                        make_unit_vector(Real3{-8, -5, 11}),
                        make_unit_vector(Real3{1, 5, 3})};

    CELER_ASSERT(axes);

    this->check_special_reflectivity_cases(axes);

    // Scan reflectivities

    std::vector<real_type> angles{
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
    };

    std::vector<real_type> te_reflectivity;
    std::vector<real_type> tm_reflectivity;
    std::vector<real_type> linear_reflectivity;

    for (real_type angle : angles)
    {
        te_reflectivity.push_back(axes.calc_reflectivity(angle, TE));
        tm_reflectivity.push_back(axes.calc_reflectivity(angle, TM));
        linear_reflectivity.push_back(axes.calc_reflectivity(angle, {3, -2}));
    }

    static real_type const expected_te_reflectivity[] = {
        0.09,
        0.093959811114961,
        0.1068867418024,
        0.13234720927357,
        0.17787352246194,
        0.2579546756256,
        0.39926995878306,
        0.6481760438648,
    };
    static real_type const expected_tm_reflectivity[] = {
        0.09,
        0.086108869837097,
        0.07429855400311,
        0.054477043367531,
        0.028097426313942,
        0.0034910285529347,
        0.015585640163587,
        0.20911860856539,
    };
    static real_type const expected_linear_reflectivity[] = {
        0.09,
        0.091544136875618,
        0.096859607094928,
        0.10838715822556,
        0.13178856980102,
        0.17965816883401,
        0.28121324536169,
        0.51308144838806,
    };

    EXPECT_VEC_SOFT_EQ(expected_te_reflectivity, te_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_tm_reflectivity, tm_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_linear_reflectivity, linear_reflectivity);
}

//---------------------------------------------------------------------------//
// Scan reflectivities for internal reflection
TEST_F(FresnelTest, internal_reflectivity)
{
    // Internal reflection has relative index < 1
    CoordinateAxes axes{2.0 / 3.0,
                        make_unit_vector(Real3{-2, 1, -1}),
                        make_unit_vector(Real3{-8, -5, 11}),
                        make_unit_vector(Real3{1, 5, 3})};

    this->check_special_reflectivity_cases(axes);

    // Critical angle implies total internal reflection
    auto critical_angle = std::asin(axes.rel_r_index);
    EXPECT_SOFT_EQ(0.99999992460542797,
                   axes.calc_reflectivity(critical_angle, TE));
    EXPECT_SOFT_EQ(0.9999998303622214,
                   axes.calc_reflectivity(critical_angle, TM));

    // Scan reflectivities

    std::vector<real_type> angles{
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
    };

    std::vector<real_type> te_reflectivity;
    std::vector<real_type> tm_reflectivity;
    std::vector<real_type> linear_reflectivity;

    for (real_type angle : angles)
    {
        te_reflectivity.push_back(axes.calc_reflectivity(angle, TE));
        tm_reflectivity.push_back(axes.calc_reflectivity(angle, TM));
        linear_reflectivity.push_back(axes.calc_reflectivity(angle, {1, 4}));
    }

    std::vector<real_type> expected_te_reflectivity{
        0.04,
        0.045207804703,
        0.0675250297305,
        0.15931858889,
        1,
        1,
        1,
        1,
    };
    std::vector<real_type> expected_tm_reflectivity{
        0.04,
        0.0350857872156,
        0.0192136283867742,
        0.000294721694597205,
        1,
        1,
        1,
        1,
    };
    std::vector<real_type> expected_linear_reflectivity{
        0.04,
        0.035681200009,
        0.0220554755246424,
        0.00964906682373,
        1,
        1,
        1,
        1,
    };

    EXPECT_VEC_SOFT_EQ(expected_te_reflectivity, te_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_tm_reflectivity, tm_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_linear_reflectivity, linear_reflectivity);
}

//---------------------------------------------------------------------------//
// Test dielectric-dielectric refracted wave calculation
TEST_F(FresnelTest, external_refracted)
{
    CoordinateAxes axes{13.0 / 7.0,
                        make_unit_vector(Real3{-2, 1, -1}),
                        make_unit_vector(Real3{-8, -5, 11}),
                        make_unit_vector(Real3{1, 5, 3})};

    CELER_ASSERT(axes);

    std::vector<real_type> angles{
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
    };

    static real_type const expected_cos_theta[] = {
        1,
        0.99426162533,
        0.977768605566,
        0.952659823628,
        0.922386317633,
        0.891459817679,
        0.864944688074,
        0.847605582097,
    };

    static real_type const expected_all_parl[] = {0, 0, 0, 0, 0, 0, 0, 0};
    static real_type const expected_all_perp[] = {1, 1, 1, 1, 1, 1, 1, 1};

    // Incident TE
    {
        auto result = scan_refraction(axes, TE, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.p_component);
    }
    // Incident TM
    {
        auto result = scan_refraction(axes, TM, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.p_component);
    }
    // Incident Linear
    {
        auto result = scan_refraction(axes, {-7, -24}, angles);

        static double const expected_s_component[] = {
            -0.28,
            -0.15234346055511,
            -0.14380497983219,
            -0.12924485161699,
            -0.10845936020105,
            -0.081863630504069,
            -0.051329677353957,
            -0.021000222305329,
        };

        static double const expected_p_component[] = {
            -0.96,
            -0.98832761270041,
            -0.98960604675571,
            -0.99161271085566,
            -0.99410088380646,
            -0.99664354008878,
            -0.99868176323729,
            -0.99977947101505,
        };

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_s_component, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_p_component, result.p_component);
    }
}

//---------------------------------------------------------------------------//
// Test dielectric-dielectric refracted wave calculation
TEST_F(FresnelTest, internal_refracted)
{
    CoordinateAxes axes{2.0 / 3.0,
                        make_unit_vector(Real3{-2, 1, -1}),
                        make_unit_vector(Real3{-8, -5, 11}),
                        make_unit_vector(Real3{1, 5, 3})};

    CELER_ASSERT(axes);

    std::vector<real_type> angles{
        0,
        0.2,
        0.4,
        0.6,
    };

    static real_type const expected_cos_theta[] = {
        1,
        0.954564622356,
        0.811661904992,
        0.53165070656,
    };

    static real_type const expected_all_parl[] = {0, 0, 0, 0};
    static real_type const expected_all_perp[] = {1, 1, 1, 1};

    // Incident TE
    {
        auto result = scan_refraction(axes, TE, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.p_component);
    }
    // Incident TM
    {
        auto result = scan_refraction(axes, TM, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.p_component);
    }
    // Incident Linear
    {
        auto result = scan_refraction(axes, {4, 3}, angles);

        static double const expected_s_component[] = {
            0.8,
            0.89814489255168,
            0.91127846614442,
            0.94349725847037,
        };
        static double const expected_p_component[] = {
            0.6,
            0.43969961562791,
            0.41179067150856,
            0.33138033022328,
        };

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_s_component, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_p_component, result.p_component);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
