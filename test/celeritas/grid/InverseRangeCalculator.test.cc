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
};

TEST_F(InverseRangeCalculatorTest, simple)
{
    // Note: these are all the same values as the RangeCalculator test.
    GridInput grid;
    grid.emin = 10;
    grid.emax = 1e4;
    grid.value = VecReal{0.5, 5, 50, 500};
    this->build(grid);

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

TEST_F(InverseRangeCalculatorTest, linear)
{
    GridInput grid;
    grid.emin = 1e-4;
    grid.emax = 1e8;
    grid.value = VecReal{
        2.38189279375507e-07,  6.207241798978842e-07, 3.33777980009005e-06,
        2.615550398212273e-05, 0.0002582189103050969, 0.00266345694155107,
        0.02296831209098076,   0.1321475316409557,    0.5688393708850199,
        2.264286285075896,     8.88454689545553,      35.09105167631849,
        139.3915036592351,     554.6294636334578,     2207.724370762173,
        8788.814443150621,     34988.60610004526,     139291.8553994672,
        554530.5699918197,     2207625.667700969,     8788715.787750119,
    };
    this->build(grid);

    InverseRangeCalculator calc_energy(this->data(), this->values());

    // Values in range
    EXPECT_SOFT_EQ(3.0402753589113166e-4, calc_energy(5e-7).value());
    EXPECT_SOFT_EQ(5.6377151632530176e-4, calc_energy(1e-6).value());
    EXPECT_SOFT_EQ(2.9643848527225914e-3, calc_energy(1e-5).value());
    EXPECT_SOFT_EQ(4.8212383419800102e-2, calc_energy(1e-3).value());
    EXPECT_SOFT_EQ(11.092870177340949, calc_energy(1).value());
    EXPECT_SOFT_EQ(11377.100982060778, calc_energy(1e3).value());
    EXPECT_SOFT_EQ(56891132.654488541, calc_energy(5e6).value());
}

TEST_F(InverseRangeCalculatorTest, spline)
{
    GridInput grid;
    grid.emin = 1e-4;
    grid.emax = 1e8;
    grid.value = VecReal{
        2.38189279375507e-07,  6.207241798978842e-07, 3.33777980009005e-06,
        2.615550398212273e-05, 0.0002582189103050969, 0.00266345694155107,
        0.02296831209098076,   0.1321475316409557,    0.5688393708850199,
        2.264286285075896,     8.88454689545553,      35.09105167631849,
        139.3915036592351,     554.6294636334578,     2207.724370762173,
        8788.814443150621,     34988.60610004526,     139291.8553994672,
        554530.5699918197,     2207625.667700969,     8788715.787750119,
    };
    this->build_spline_inverse(grid, BC::not_a_knot);

    InverseRangeCalculator calc_energy(this->data(), this->values());

    // Values in range
    EXPECT_SOFT_EQ(3.0914474675693040e-4, calc_energy(5e-7).value());
    EXPECT_SOFT_EQ(6.4951208258105981e-4, calc_energy(1e-6).value());
    EXPECT_SOFT_EQ(3.3028905236727659e-3, calc_energy(1e-5).value());
    EXPECT_SOFT_EQ(5.2562387201304524e-2, calc_energy(1e-3).value());
    EXPECT_SOFT_EQ(10.959516048098248, calc_energy(1).value());
    EXPECT_SOFT_EQ(11378.157574904253, calc_energy(1e3).value());
    EXPECT_SOFT_EQ(56891307.88507662, calc_energy(5e6).value());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
