//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/PhysicsGridCalculator.hh"

#include <algorithm>
#include <cmath>
#include "base/Interpolator.hh"
#include "base/PieBuilder.hh"
#include "base/Range.hh"
#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PhysicsGridCalculatorTest : public celeritas::Test
{
  protected:
    using Energy    = PhysicsGridCalculator::Energy;
    using real_type = celeritas::real_type;

    void build(real_type emin, real_type emax, int count)
    {
        CELER_EXPECT(count >= 2);
        data.log_energy = UniformGridData::from_bounds(
            std::log(emin), std::log(emax), count);

        std::vector<real_type> temp_xs(count);

        // Interpolate xs grid: linear in bin, log in energy
        using celeritas::Interp;
        celeritas::Interpolator<Interp::linear, Interp::log, real_type> calc_xs(
            {0.0, emin}, {count - 1.0, emax});
        for (auto i : celeritas::range(temp_xs.size()))
        {
            temp_xs[i] = calc_xs(i);
        }

        data.value = make_pie_builder(&stored_xs)
                         .insert_back(temp_xs.begin(), temp_xs.end());

        CELER_ENSURE(data);
        CELER_ENSURE(celeritas::soft_equal(emax, stored_xs[data.value].back()));
    }

    XsGridData                                       data;
    Pie<real_type, Ownership::value, MemSpace::host> stored_xs;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhysicsGridCalculatorTest, simple)
{
    // Energy from 1 to 1e5 MeV with 5 grid points; XS should be the same
    // *No* magical 1/E scaling
    this->build(1.0, 1e5, 6);

    PhysicsGridCalculator calc(this->data, this->stored_xs);

    // Test on grid points
    EXPECT_SOFT_EQ(1.0, calc(Energy{1}));
    EXPECT_SOFT_EQ(1e2, calc(Energy{1e2}));
    EXPECT_SOFT_EQ(1e5 - 1e-6, calc(Energy{1e5 - 1e-6}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e5}));

    // Test between grid points
    EXPECT_SOFT_EQ(5, calc(Energy{5}));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(1.0, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e7}));
}

TEST_F(PhysicsGridCalculatorTest, scaled_lowest)
{
    // Energy from .1 to 1e4 MeV with 5 grid points; XS should be constant
    // since the constructor fills it with E
    this->build(0.1, 1e4, 6);
    data.prime_index = 0;

    PhysicsGridCalculator calc(this->data, this->stored_xs);

    // Test on grid points
    EXPECT_SOFT_EQ(1, calc(Energy{0.1}));
    EXPECT_SOFT_EQ(1, calc(Energy{1e2}));
    EXPECT_SOFT_EQ(1, calc(Energy{1e4 - 1e-6}));
    EXPECT_SOFT_EQ(1, calc(Energy{1e4}));

    // Test between grid points
    EXPECT_SOFT_EQ(1, calc(Energy{0.2}));
    EXPECT_SOFT_EQ(1, calc(Energy{5}));

    // Test out-of-bounds: cross section still scales according to 1/E (TODO:
    // this might not be the best behavior for the lower energy value)
    EXPECT_SOFT_EQ(1000, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(0.1, calc(Energy{1e5}));
}

TEST_F(PhysicsGridCalculatorTest, scaled_middle)
{
    // Energy from .1 to 1e4 MeV with 5 grid points; XS should be constant
    // since the constructor fills it with E
    this->build(0.1, 1e4, 6);
    data.prime_index = 3;
    auto xs          = this->stored_xs[data.value];
    std::fill(xs.begin(), xs.begin() + 3, 1.0);

    // Change constant to 3 just to shake things up
    for (real_type& xs : xs)
    {
        xs *= 3;
    }

    PhysicsGridCalculator calc(this->data, this->stored_xs);

    // Test on grid points
    EXPECT_SOFT_EQ(3, calc(Energy{0.1}));
    EXPECT_SOFT_EQ(3, calc(Energy{1e2}));
    EXPECT_SOFT_EQ(3, calc(Energy{1e4 - 1e-6}));
    EXPECT_SOFT_EQ(3, calc(Energy{1e4}));

    // Test between grid points
    EXPECT_SOFT_EQ(3, calc(Energy{0.2}));
    EXPECT_SOFT_EQ(3, calc(Energy{5}));

    // Test out-of-bounds: cross section still scales according to 1/E (TODO:
    // this might not be the right behavior for
    EXPECT_SOFT_EQ(3, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(0.3, calc(Energy{1e5}));
}

TEST_F(PhysicsGridCalculatorTest, scaled_highest)
{
    // values of 1, 10, 100 --> actual xs = {1, 10, 1}
    this->build(1, 100, 3);
    data.prime_index = 2;

    PhysicsGridCalculator calc(this->data, this->stored_xs);
    EXPECT_SOFT_EQ(1, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(1, calc(Energy{1}));
    EXPECT_SOFT_EQ(10, calc(Energy{10}));
    EXPECT_SOFT_EQ(2.0, calc(Energy{90}));

    // Final point and higher are scaled by 1/E
    EXPECT_SOFT_EQ(1, calc(Energy{100}));
    EXPECT_SOFT_EQ(.1, calc(Energy{1000}));
}

TEST_F(PhysicsGridCalculatorTest, TEST_IF_CELERITAS_DEBUG(scaled_off_the_end))
{
    // values of 1, 10, 100 --> actual xs = {1, 10, 100}
    this->build(1, 100, 3);
    data.prime_index = 3; // disallowed

    EXPECT_THROW(PhysicsGridCalculator(this->data, this->stored_xs),
                 celeritas::DebugError);
}
