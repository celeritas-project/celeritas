//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseRangeCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/InverseRangeCalculator.hh"

#include "corecel/math/SoftEqual.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class InverseRangeCalculatorTest : public CalculatorTestBase
{
  protected:
    using Energy = InverseRangeCalculator::Energy;

    void SetUp() override
    {
        // Energy from 1e1 to 1e4 MeV with 3 bins (4 points)
        this->build(10, 1e4, 4);

        // InverseRange is 1/20 of energy
        auto value_span = this->mutable_values();
        for (real_type& xs : value_span)
        {
            xs *= .05;
        }

        // Adjust final point for roundoff for exact top-of-range testing
        CELER_ASSERT(soft_equal(real_type(500), value_span.back()));
        value_span.back() = 500;
    }
};

// Note: these are all the same values as the RangeCalculator test.
TEST_F(InverseRangeCalculatorTest, all)
{
    InverseRangeCalculator calc_energy(this->data(), this->values());

    // Values below should be scaled below emin
    EXPECT_SOFT_EQ(1.0, calc_energy(.5 * std::sqrt(1. / 10.)).value());
    EXPECT_SOFT_EQ(2.0, calc_energy(.5 * std::sqrt(2. / 10.)).value());

    // Values in range
    EXPECT_SOFT_EQ(10.0, calc_energy(.5).value());
    EXPECT_SOFT_EQ(20.0, calc_energy(1).value());
    EXPECT_SOFT_EQ(100.0, calc_energy(5).value());

    // Top of range
    EXPECT_SOFT_EQ(1e4, calc_energy(500).value());

#if CELERITAS_DEBUG
    // Above range
    EXPECT_THROW(calc_energy(500.1), DebugError);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
