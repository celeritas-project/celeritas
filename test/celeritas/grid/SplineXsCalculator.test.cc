//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineXsCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/SplineXsCalculator.hh"

#include <algorithm>
#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SplineXsCalculatorTest : public CalculatorTestBase
{
  protected:
    using Energy = SplineXsCalculator::Energy;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SplineXsCalculatorTest, simple)
{
    // Energy from 1 to 1e5 MeV with 5 grid points; XS should be the same
    // *No* magical 1/E scaling
    this->build(1.0, 1e5, 6);

    for (size_type order = 1; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on grid points
        EXPECT_SOFT_EQ(1.0, calc(Energy{1}));
        EXPECT_SOFT_EQ(1e2, calc(Energy{1e2}));
        EXPECT_SOFT_EQ(1e5 - 1e-6, calc(Energy{1e5 - 1e-6}));
        EXPECT_SOFT_EQ(1e5, calc(Energy{1e5}));

        // Test access by index
        EXPECT_SOFT_EQ(1.0, calc[0]);
        EXPECT_SOFT_EQ(1e2, calc[2]);
        EXPECT_SOFT_EQ(1e5, calc[5]);

        // Test between grid points
        EXPECT_SOFT_EQ(5, calc(Energy{5}));
        EXPECT_SOFT_EQ(5e2, calc(Energy{5e2}));
        EXPECT_SOFT_EQ(5e4, calc(Energy{5e4}));

        // Test out-of-bounds
        EXPECT_SOFT_EQ(1.0, calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(1e5, calc(Energy{1e7}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1.0, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e5, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, scaled_lowest)
{
    // Energy from .1 to 1e4 MeV with 5 grid points; XS should be constant
    // since the constructor fills it with E
    this->build(0.1, 1e4, 6);
    this->set_prime_index(0);

    for (size_type order = 1; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on grid points
        EXPECT_SOFT_EQ(1, calc(Energy{0.1}));
        EXPECT_SOFT_EQ(1, calc(Energy{1e2}));
        EXPECT_SOFT_EQ(1, calc(Energy{1e4 - 1e-6}));
        EXPECT_SOFT_EQ(1, calc(Energy{1e4}));

        // Test access by index
        EXPECT_SOFT_EQ(1, calc[0]);
        EXPECT_SOFT_EQ(1, calc[2]);
        EXPECT_SOFT_EQ(1, calc[5]);

        // Test between grid points
        EXPECT_SOFT_EQ(1, calc(Energy{0.2}));
        EXPECT_SOFT_EQ(1, calc(Energy{5}));
        EXPECT_SOFT_EQ(1, calc(Energy{2e3}));

        // Test out-of-bounds: cross section still scales according to 1/E
        // (TODO: this might not be the best behavior for the lower energy
        // value)
        EXPECT_SOFT_EQ(1000, calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(0.1, calc(Energy{1e5}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(0.1, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, scaled_middle)
{
    // Energy from .1 to 1e4 MeV with 5 grid points; XS should be constant
    // since the constructor fills it with E
    this->build(0.1, 1e4, 6);
    this->set_prime_index(3);
    auto xs = this->mutable_values();
    std::fill(xs.begin(), xs.begin() + 3, 1.0);

    // Change constant to 3 just to shake things up
    for (real_type& x : xs)
    {
        x *= 3;
    }

    for (size_type order = 1; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on grid points
        EXPECT_SOFT_EQ(3, calc(Energy{0.1}));
        EXPECT_SOFT_EQ(3, calc(Energy{1e2}));
        EXPECT_SOFT_EQ(3, calc(Energy{1e4 - 1e-6}));
        EXPECT_SOFT_EQ(3, calc(Energy{1e4}));

        // Test access by index
        EXPECT_SOFT_EQ(3, calc[0]);
        EXPECT_SOFT_EQ(3, calc[2]);
        EXPECT_SOFT_EQ(3, calc[5]);

        // Test between grid points
        EXPECT_SOFT_EQ(3, calc(Energy{0.2}));
        EXPECT_SOFT_EQ(3, calc(Energy{5}));
        EXPECT_SOFT_EQ(3, calc(Energy{2e3}));

        // Test out-of-bounds: cross section still scales according to 1/E
        // (TODO: this might not be the right behavior for
        EXPECT_SOFT_EQ(3, calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(0.3, calc(Energy{1e5}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(0.1, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, scaled_highest)
{
    // values of 1, 10, 100 --> actual xs = {1, 10, 1}
    this->build(1, 100, 3);
    this->set_prime_index(2);

    size_type order = 1;

    SplineXsCalculator calc(this->data(), this->values(), order);
    EXPECT_SOFT_EQ(1, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(1, calc(Energy{1}));
    EXPECT_SOFT_EQ(10, calc(Energy{10}));
    EXPECT_SOFT_EQ(2.0, calc(Energy{90}));

    // Test access by index
    EXPECT_SOFT_EQ(1, calc[0]);
    EXPECT_SOFT_EQ(10, calc[1]);
    EXPECT_SOFT_EQ(1, calc[2]);

    // Final point and higher are scaled by 1/E
    EXPECT_SOFT_EQ(1, calc(Energy{100}));
    EXPECT_SOFT_EQ(.1, calc(Energy{1000}));

    // Test energy grid bounds
    EXPECT_SOFT_EQ(1, value_as<Energy>(calc.energy_min()));
    EXPECT_SOFT_EQ(100, value_as<Energy>(calc.energy_max()));
}

TEST_F(SplineXsCalculatorTest, quadratic_simple)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.1} * ipow<2>(energy); };

    this->build({1e-3, 1e2}, 6, reference_xs);

    for (size_type order = 2; order < 5; ++order)
    {
        SCOPED_TRACE("order=" + std::to_string(order));
        SplineXsCalculator calc(this->data(), this->values(), order);

        for (real_type e : {1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2})
        {
            // Interpolation in the construction means small failures in
            // single-precision mode
            EXPECT_SOFT_NEAR(reference_xs(e), calc(Energy{e}), coarse_eps);
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc[0]);
        EXPECT_SOFT_EQ(reference_xs(1e-1), calc[2]);
        EXPECT_SOFT_EQ(reference_xs(1e2), calc[5]);

        // Test out-of-bounds
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc(Energy{1e-5}));
        EXPECT_SOFT_EQ(reference_xs(1e2), calc(Energy{1e5}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e2, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, quadratic_scaled_lowest)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.1} * ipow<2>(energy); };

    this->build({1.0e-3, 1e2}, 6, reference_xs);
    this->convert_to_prime(0);
    for (size_type order = 2; order < 5; ++order)
    {
        SCOPED_TRACE("order=" + std::to_string(order));
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e}));
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc[0]);
        EXPECT_SOFT_EQ(reference_xs(1e-1), calc[2]);
        EXPECT_SOFT_EQ(reference_xs(1e2), calc[5]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e2, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, quadratic_scaled_middle)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.1} * ipow<2>(energy); };

    this->build({1.0e-3, 1e2}, 6, reference_xs);
    this->convert_to_prime(3);
    for (size_type order = 2; order < 5; ++order)
    {
        SCOPED_TRACE("order=" + std::to_string(order));
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e}));
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc[0]);
        EXPECT_SOFT_EQ(reference_xs(1e-1), calc[2]);
        EXPECT_SOFT_EQ(reference_xs(1e2), calc[5]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e2, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, quadratic_scaled_highest)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.1} * ipow<2>(energy); };

    this->build({1.0e-3, 1e2}, 6, reference_xs);
    this->convert_to_prime(5);
    for (size_type order = 2; order < 5; ++order)
    {
        SCOPED_TRACE("order=" + std::to_string(order));
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e}));
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc[0]);
        EXPECT_SOFT_EQ(reference_xs(1e-1), calc[2]);
        EXPECT_SOFT_EQ(reference_xs(1e2), calc[5]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e2, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, cubic_simple)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.01} * ipow<3>(energy); };

    this->build({1e-3, 1e4}, 8, reference_xs);

    for (size_type order = 3; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        for (real_type e : {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 1e2, 5e2, 1e3})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e})) << "e=" << repr(e);
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-2), calc[1]);
        EXPECT_SOFT_EQ(reference_xs(1.0), calc[3]);
        EXPECT_SOFT_EQ(reference_xs(1e4), calc[7]);

        // Test out-of-bounds
        EXPECT_SOFT_EQ(reference_xs(1e-3), calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(reference_xs(1e4), calc(Energy{1e7}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, cubic_scaled_lowest)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.01} * ipow<3>(energy); };

    this->build({1e-3, 1e4}, 8, reference_xs);
    this->convert_to_prime(0);
    for (size_type order = 3; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 1e2, 5e2, 1e3})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e})) << "e=" << repr(e);
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-2), calc[1]);
        EXPECT_SOFT_EQ(reference_xs(1.0), calc[3]);
        EXPECT_SOFT_EQ(reference_xs(1e4), calc[7]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, cubic_scaled_middle)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.01} * ipow<3>(energy); };

    this->build({1e-3, 1e4}, 8, reference_xs);
    this->convert_to_prime(4);
    for (size_type order = 3; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 1e2, 5e2, 1e3})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e})) << "e=" << repr(e);
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-2), calc[1]);
        EXPECT_SOFT_EQ(reference_xs(1.0), calc[3]);
        EXPECT_SOFT_EQ(reference_xs(1e4), calc[7]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, cubic_scaled_highest)
{
    auto reference_xs
        = [](real_type energy) { return real_type{0.1} * ipow<3>(energy); };

    this->build({1e-3, 1e4}, 8, reference_xs);
    this->convert_to_prime(7);
    for (size_type order = 3; order < 5; ++order)
    {
        SplineXsCalculator calc(this->data(), this->values(), order);

        // Test on and between grid points
        for (real_type e : {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 1e2, 5e2, 1e3})
        {
            EXPECT_SOFT_EQ(reference_xs(e), calc(Energy{e})) << "e=" << repr(e);
        }

        // Test access by index
        EXPECT_SOFT_EQ(reference_xs(1e-2), calc[1]);
        EXPECT_SOFT_EQ(reference_xs(1.0), calc[3]);
        EXPECT_SOFT_EQ(reference_xs(1e4), calc[7]);

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineXsCalculatorTest, TEST_IF_CELERITAS_DEBUG(scaled_off_the_end))
{
    // values of 1, 10, 100 --> actual xs = {1, 10, 100}
    this->build(1, 100, 3);
    XsGridData data(this->data());
    data.prime_index = 3;  // disallowed

    size_type order = 2;

    EXPECT_THROW(SplineXsCalculator(data, this->values(), order), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
