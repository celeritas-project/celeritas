//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ArrayUtils.test.cc
//---------------------------------------------------------------------------//
#include "base/ArrayUtils.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/Constants.hh"

using celeritas::array;

enum
{
    X = 0,
    Y = 1,
    Z = 2
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ArrayUtilsTest, axpy)
{
    array<int, 3> x{1, 3, 2};
    array<int, 3> y{20, 30, 40};

    celeritas::axpy(4, x, &y);
    EXPECT_EQ(4 * 1 + 20, y[X]);
    EXPECT_EQ(4 * 3 + 30, y[Y]);
    EXPECT_EQ(4 * 2 + 40, y[Z]);
}

TEST(ArrayUtilsTest, dot_product)
{
    array<int, 2> x{1, 3};
    array<int, 2> y{2, 4};

    EXPECT_EQ(1 * 2 + 3 * 4, celeritas::dot_product(x, y));
}

TEST(ArrayUtilsTest, norm)
{
    EXPECT_SOFT_EQ(std::sqrt(4.0 + 9.0 + 16.0),
                   celeritas::norm(array<double, 3>{2, 3, 4}));
}

TEST(ArrayUtilsTest, normalize_direction)
{
    array<double, 3> direction{1, 2, 3};
    double           norm = 1 / std::sqrt(1 + 4 + 9);
    celeritas::normalize_direction(&direction);

    static const double expected[] = {1 * norm, 2 * norm, 3 * norm};
    EXPECT_VEC_SOFT_EQ(expected, direction);
}

TEST(ArrayUtilsTest, rotate_polar)
{
    array<double, 3> vec = {-1.1, 2.3, 0.9};
    celeritas::normalize_direction(&vec);

    // transform through some directions
    const double costheta = std::cos(2.0 / 3.0);
    const double sintheta = std::sqrt(1.0 - costheta * costheta);
    const double phi      = celeritas::constants::two_pi / 3.0;

    double           a = 1.0 / sqrt(1.0 - vec[Z] * vec[Z]);
    array<double, 3> expected
        = {vec[X] * costheta + vec[Z] * vec[X] * sintheta * cos(phi) * a
               - vec[Y] * sintheta * sin(phi) * a,
           vec[Y] * costheta + vec[Z] * vec[Y] * sintheta * cos(phi) * a
               + vec[X] * sintheta * sin(phi) * a,
           vec[Z] * costheta - sintheta * cos(phi) / a};
    celeritas::rotate_polar(costheta, phi, &vec);
    EXPECT_VEC_SOFT_EQ(expected, vec);

    // Transform degenerate vector along y
    expected = {sintheta * cos(phi), sintheta * sin(phi), -costheta};
    vec      = {0.0, 0.0, -1.0};
    celeritas::rotate_polar(costheta, phi, &vec);
    EXPECT_VEC_SOFT_EQ(expected, vec);

    expected = {sintheta * cos(phi), sintheta * sin(phi), costheta};
    vec      = {0.0, 0.0, 1.0};
    celeritas::rotate_polar(costheta, phi, &vec);
    EXPECT_VEC_SOFT_EQ(expected, vec);
}
