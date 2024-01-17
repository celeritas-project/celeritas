//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/RangeCalculator.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class RangeCalculatorTest : public CalculatorTestBase
{
  protected:
    using Energy = RangeCalculator::Energy;

    void SetUp() override
    {
        // Energy from 1e1 to 1e4 MeV with 3 bins (4 points)
        this->build(10, 1e4, 4);

        // Range is 1/20 of energy
        for (real_type& xs : this->mutable_values())
        {
            xs *= .05;
        }
    }
};

TEST_F(RangeCalculatorTest, all)
{
    RangeCalculator calc_range(this->data(), this->values());

    // "stopped" particle case should not calculate range
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_range(Energy{0}), DebugError);
    }

    // Values below should be scaled below emin
    EXPECT_SOFT_EQ(.5 * std::sqrt(1. / 10.), calc_range(Energy{1}));
    EXPECT_SOFT_EQ(.5 * std::sqrt(2. / 10.), calc_range(Energy{2}));
    // Values in range
    EXPECT_SOFT_EQ(0.5, calc_range(Energy{10}));
    EXPECT_SOFT_EQ(1.0, calc_range(Energy{20}));
    EXPECT_SOFT_EQ(5.0, calc_range(Energy{100}));

    // Top of range
    EXPECT_SOFT_EQ(500, calc_range(Energy{1e4}));
    // Above range
    EXPECT_SOFT_EQ(500, calc_range(Energy{1.001e4}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
