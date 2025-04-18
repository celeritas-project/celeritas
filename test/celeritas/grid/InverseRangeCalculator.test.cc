//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseRangeCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/InverseRangeCalculator.hh"

#include <vector>

#include "corecel/math/SoftEqual.hh"
#include "celeritas/grid/RangeCalculator.hh"

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
    using VecReal = std::vector<real_type>;
};

TEST_F(InverseRangeCalculatorTest, simple)
{
    // Note: these are all the same values as the RangeCalculator test.
    inp::UniformGrid grid;
    grid.x = {10, 1e4};
    grid.y = {0.5, 5, 50, 500};
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

TEST_F(InverseRangeCalculatorTest, interpolation)
{
    // Trimmed range table values
    inp::UniformGrid grid;
    grid.x = {1e-4, 1e8};
    grid.y = {
        2.38189279375507e-07,  6.207241798978842e-07, 3.33777980009005e-06,
        2.615550398212273e-05, 0.0002582189103050969, 0.00266345694155107,
        0.02296831209098076,   0.1321475316409557,    0.5688393708850199,
        2.264286285075896,     8.88454689545553,      35.09105167631849,
        139.3915036592351,     554.6294636334578,     2207.724370762173,
        8788.814443150621,     34988.60610004526,     139291.8553994672,
        554530.5699918197,     2207625.667700969,     8788715.787750119,
    };

    VecReal range{5e-7, 1e-6, 1e-5, 1e-3, 1, 1e3, 5e6};
    {
        // Test linear interpolation
        real_type tol = SoftEqual{}.rel();
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            tol *= 10;
        }

        VecReal energy{
            3.0402753589113166e-4,
            5.6377151632530176e-4,
            2.9643848527225914e-3,
            4.8212383419800102e-2,
            11.092870177340949,
            11377.100982060778,
            56891132.654488541,
        };

        this->build(grid);
        InverseRangeCalculator calc_energy(this->data(), this->values());
        for (size_type i = 0; i < range.size(); ++i)
        {
            EXPECT_SOFT_NEAR(
                energy[i], value_as<Energy>(calc_energy(range[i])), tol);
        }

        // Linear interpolation is invertible
        RangeCalculator calc_range(this->data(), this->values());
        for (size_type i = 0; i < range.size(); ++i)
        {
            EXPECT_SOFT_NEAR(range[i], calc_range(Energy(energy[i])), tol);
        }
    }
    {
        // Test cubic spline interpolation
        grid.interpolation.type = InterpolationType::cubic_spline;
        grid.interpolation.bc = BC::not_a_knot;

        VecReal energy{
            3.0914474675693040e-4,
            6.4951208258105981e-4,
            3.3028905236727659e-3,
            5.2562387201304524e-2,
            10.959516048098248,
            11378.157574904253,
            56891307.88507662,
        };

        this->build_inverted(grid);
        InverseRangeCalculator calc_energy(this->data(), this->values());
        for (size_type i = 0; i < range.size(); ++i)
        {
            EXPECT_SOFT_EQ(energy[i], value_as<Energy>(calc_energy(range[i])));
        }

        // Spline interpolation is not necessarily invertible
        VecReal roundtrip_range{
            4.9177721122147e-07,
            1.0488196731322e-06,
            9.6171528949346e-06,
            0.00090357562168923,
            0.98932838203235,
            1000.1338772112,
            5000022.2149638,
        };

        this->build(grid);
        RangeCalculator calc_range(this->data(), this->values());
        for (size_type i = 0; i < range.size(); ++i)
        {
            EXPECT_SOFT_EQ(roundtrip_range[i], calc_range(Energy(energy[i])));
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
