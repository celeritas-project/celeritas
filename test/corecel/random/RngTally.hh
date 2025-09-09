//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/RngTally.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/math/Algorithms.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Tally RNG moments and check them.
 */
struct RngTally
{
    double moments[4] = {0, 0, 0, 0};
    double min = 1;
    double max = 0;

    void operator()(double xi)
    {
        this->min = std::min(xi, this->min);
        this->max = std::max(xi, this->max);

        // Rescale to [-1, 1]
        xi = 2 * xi - 1;
        this->moments[0] += xi;
        this->moments[1] += 0.5 * (3 * ipow<2>(xi) - 1);
        this->moments[2] += 0.5 * (5 * ipow<3>(xi) - 3 * xi);
        this->moments[3] += 0.125 * (35 * ipow<4>(xi) - 30 * ipow<2>(xi) + 3);
    }

    void check(double num_samples, double tol)
    {
        CELER_EXPECT(tol < 1);

        EXPECT_LT(max, 1);
        EXPECT_GE(min, 0);

        for (auto& m : this->moments)
        {
            m /= num_samples;
        }
        EXPECT_NEAR(0, moments[0], tol);
        EXPECT_NEAR(0, moments[1], tol);
        EXPECT_NEAR(0, moments[2], tol);
        EXPECT_NEAR(0, moments[3], tol);
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
