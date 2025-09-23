//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/DielectricDielectricCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/model/DielectricDielectricCalculator.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/optical/surface/model/FresnelReflectivityCalculator.hh"

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

struct LinearPolarization
{
    real_type t_e;
    real_type t_m;
};

static LinearPolarization const TE{1, 0};
static LinearPolarization const TM{0, 1};

struct CoordinateAxes
{
    real_type rel_r_index;

    Real3 n_hat;
    Real3 s_hat;
    Real3 p_hat;

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
        return FresnelReflectivityCalculator{
            PhotonPhasor{this->make_direction(angle),
                         this->make_polarization(angle, pol)},
            n_hat,
            rel_r_index}();
    }

    SurfaceInteraction
    calc_dielectric_dielectric(real_type angle,
                               LinearPolarization const& pol) const
    {
        return DielectricDielectricCalculator{
            PhotonPhasor{this->make_direction(angle),
                         this->make_polarization(angle, pol)},
            n_hat,
            rel_r_index}();
    }
};

struct ScatteringResult
{
    std::vector<real_type> cos_theta;
    std::vector<real_type> s_component;
    std::vector<real_type> p_component;
};

ScatteringResult
scan_dielectric_dielectric(CoordinateAxes const& axes,
                           LinearPolarization const& pol,
                           std::vector<real_type> const& angles)
{
    ScatteringResult result;

    for (auto angle : angles)
    {
        auto refract = axes.calc_dielectric_dielectric(angle, pol);

        EXPECT_EQ(SurfaceInteraction::refract, refract.action);
        EXPECT_TRUE(refract.photon);
        EXPECT_SOFT_EQ(0, dot_product(refract.photon.direction, axes.p_hat));

        real_type cos_theta
            = -dot_product(refract.photon.direction, axes.n_hat);
        real_type theta = std::acos(cos_theta);

        result.cos_theta.push_back(cos_theta);
        result.s_component.push_back(dot_product(
            refract.photon.polarization, axes.make_polarization(theta, TE)));
        result.p_component.push_back(dot_product(
            refract.photon.polarization, axes.make_polarization(theta, TM)));
    }

    return result;
}

//---------------------------------------------------------------------------//
// TEST CHASIS
//---------------------------------------------------------------------------//

class DielectricDielectricCalculatorTest : public ::celeritas::test::Test
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
TEST_F(DielectricDielectricCalculatorTest, external_reflectivity)
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

    std::vector<real_type> expected_te_reflectivity{
        0.09,
        0.093959811115,
        0.106886741802,
        0.132347209274,
        0.177876522462,
        0.257954675626,
        0.399269958783,
        0.648176043865,
    };
    std::vector<real_type> expected_tm_reflectivity{
        0.09,
        0.0861088698371,
        0.0742985540031,
        0.0544770433675,
        0.0280974263139,
        0.00349102855293,
        0.0155856401636,
        0.209118608565,
    };
    std::vector<real_type> expected_linear_reflectivity{
        0.09,
        0.0915441368756,
        0.0968596070949,
        0.108387158226,
        0.131788569801,
        0.179658168834,
        0.281213245362,
        0.513081448388,
    };

    EXPECT_VEC_SOFT_EQ(expected_te_reflectivity, te_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_tm_reflectivity, tm_reflectivity);
    EXPECT_VEC_SOFT_EQ(expected_linear_reflectivity, linear_reflectivity);
}

//---------------------------------------------------------------------------//
// Scan reflectivities for internal reflection
TEST_F(DielectricDielectricCalculatorTest, internal_reflectivity)
{
    // Internal reflection has relative index < 1
    CoordinateAxes axes{2.0 / 3.0,
                        make_unit_vector(Real3{-2, 1, -1}),
                        make_unit_vector(Real3{-8, -5, 11}),
                        make_unit_vector(Real3{1, 5, 3})};

    this->check_special_reflectivity_cases(axes);

    // Critical angle implies total internal reflection
    auto critical_angle = std::asin(axes.rel_r_index);
    EXPECT_SOFT_EQ(1, axes.calc_reflectivity(critical_angle, TE));
    EXPECT_SOFT_EQ(1, axes.calc_reflectivity(critical_angle, TM));

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
        0.0192136283868,
        0.000294721684597,
        1,
        1,
        1,
        1,
    };
    std::vector<real_type> expected_linear_reflectivity{
        0.04,
        0.035681200009,
        0.0220554755246,
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
TEST_F(DielectricDielectricCalculatorTest, external_refracted)
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
        auto result = scan_dielectric_dielectric(axes, TE, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.p_component);
    }
    // Incident TM
    {
        auto result = scan_dielectric_dielectric(axes, TM, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.p_component);
    }
    // Incident Linear
    {
        auto result = scan_dielectric_dielectric(axes, {-7, -24}, angles);

        real_type s = -7.0 / 25.0;
        real_type p = -24 / 25.0;

        static real_type const expected_s_component[]
            = {s, s, s, s, s, s, s, s};
        static real_type const expected_p_component[]
            = {p, p, p, p, p, p, p, p};

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_s_component, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_p_component, result.p_component);
    }
}

//---------------------------------------------------------------------------//
// Test dielectric-dielectric refracted wave calculation
TEST_F(DielectricDielectricCalculatorTest, internal_refracted)
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
        auto result = scan_dielectric_dielectric(axes, TE, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.p_component);
    }
    // Incident TM
    {
        auto result = scan_dielectric_dielectric(axes, TM, angles);

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_all_parl, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_all_perp, result.p_component);
    }
    // Incident Linear
    {
        auto result = scan_dielectric_dielectric(axes, {4, 3}, angles);

        static real_type const expected_s_component[] = {0.8, 0.8, 0.8, 0.8};
        static real_type const expected_p_component[] = {0.6, 0.6, 0.6, 0.6};

        EXPECT_VEC_SOFT_EQ(expected_cos_theta, result.cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_s_component, result.s_component);
        EXPECT_VEC_SOFT_EQ(expected_p_component, result.p_component);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
