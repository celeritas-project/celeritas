//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
        inp::UniformGrid grid;
        grid.x = {10, 1e4};
        grid.y = {0.5, 5, 50, 500};
        this->build(grid);
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

    if (CELERITAS_DEBUG)
    {
        // Above range
        EXPECT_THROW(calc_energy(500.1), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
