//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/RangeGridCalculator.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Tabulated energy loss in steel as a funtion of energy from 0.1 keV--100 TeV.
 *
 * These values and the expected range are the imported values from Geant4 for
 * the electron ionization process taken from four-steel-slabs.root.
 */
Span<real_type const> get_dedx()
{
    // On uniform log energy grid from 1e-4 to 1e8 MeV
    static real_type const dedx[] = {
        839.668353354807, 844.813525381296, 827.450592332634, 789.80588282839,
        705.134542622882, 598.202746440131, 507.487712297195, 430.530096695467,
        368.284055543883, 315.470283756288, 263.57841571475,  216.264734170156,
        175.063530384051, 140.289245825116, 111.600220710967, 88.3413678229782,
        69.7481174579048, 55.0382090170905, 43.531629942981,  34.6222581345934,
        27.7915893876484, 22.6117194229536, 18.7364170928593, 15.8890486367146,
        13.8519448788249, 12.4539689722886, 11.5250987040272, 10.9449849886648,
        10.6619173294951, 10.5972574381541, 10.6855079931534, 10.8468742900639,
        10.8067863342013, 10.8511175046,    10.9250966242212, 11.0069268409596,
        11.0860701208148, 11.1571249012729, 11.2177781962844, 11.2674206922356,
        11.3062721121368, 11.335174675971,  11.3553238163283, 11.3681915302876,
        11.3754104300348, 11.3786030042885, 11.3793047984112, 11.3789382701874,
        11.3786063588217, 11.3784262549454, 11.3783308940568, 11.3782808891513,
        11.3782548371818, 11.3782413084296, 11.3782342940126, 11.3782306598131,
        11.378228777509,  11.3782278026977, 11.3782272978736, 11.3782270364384,
        11.3782269010447, 11.3782268309245, 11.3782267946085, 11.3782267757997,
        11.3782267660582, 11.3782267610127, 11.3782267583995, 11.378226757046,
        11.378226756345,  11.3782267559818, 11.3782267557938, 11.3782267556964,
        11.3782267556459, 11.3782267556198, 11.3782267556063, 11.3782267555993,
        11.3782267555956, 11.3782267555937, 11.3782267555928, 11.3782267555923,
        11.378226755592,  11.3782267555919, 11.3782267555918, 11.3782267555918,
        11.3782267555917};
    return make_span(dedx);
}

//---------------------------------------------------------------------------//

class RangeGridCalculatorTest : public Test
{
  protected:
    using BC = SplineDerivCalculator::BoundaryCondition;

    //! Build a dE/dx grid from energy bounds and values
    void build(real_type energy_min,
               real_type energy_max,
               Span<real_type const> dedx)
    {
        dedx_grid.x = {std::log(energy_min), std::log(energy_max)};
        dedx_grid.y = {dedx.begin(), dedx.end()};
        CELER_ENSURE(dedx_grid);
    }

    inp::UniformGrid dedx_grid;
};

TEST_F(RangeGridCalculatorTest, geant)
{
    this->build(1e-4, 1e8, get_dedx());

    auto range = RangeGridCalculator(BC::geant)(dedx_grid);

    // Range values imported from Geant4
    static double const expected_range[] = {
        2.38189279375507e-07, 2.84408954171739e-07, 3.49102977951787e-07,
        4.41948426764455e-07, 5.81556823586e-07,    8.05901114436151e-07,
        1.17454066145004e-06, 1.77745754547844e-06, 2.76204861844086e-06,
        4.35667846840664e-06, 6.97442596899499e-06, 1.13782785072204e-05,
        1.88891639349819e-05, 3.18605823919917e-05, 5.44434027717438e-05,
        9.40070870338402e-05, 0.000163563219143859, 0.000286037892868613,
        0.000501482878278527, 0.00087893144157381,  0.00153525772246054,
        0.00266345694155107,  0.00457191995334743,  0.00773349334255941,
        0.0128386852149239,   0.0208425935522307,   0.033018366007371,
        0.0510471467941446,   0.0770689461686799,   0.113770052130649,
        0.16467102523814,     0.234464878004049,    0.330849363091667,
        0.465100966357566,    0.650288316035434,    0.905910815344281,
        1.25839303290099,     1.74491327929517,     2.41692822783202,
        3.34615488778155,     4.63227661577805,     6.41405420623578,
        8.88454689545553,     12.312401454691,      17.0713057583066,
        23.6808998605494,     32.8634861595932,     45.6226216547902,
        63.3520195734745,     87.9874473250438,     122.218669510346,
        169.783079455779,     235.873826664002,     327.706770777358,
        455.308343579491,     632.610231666306,     878.970463443993,
        1221.2869358411,      1696.93416138258,     2357.84385913124,
        3276.17492158059,     4552.19180606389,     6325.21152567767,
        8788.81444315062,     12211.9796009805,     16968.4521672013,
        23577.5493691843,     32760.8601547275,     45521.0291157603,
        63251.2263953397,     87887.2556302274,     122118.907251755,
        169683.632945113,     235774.604987336,     327607.712858899,
        455209.402480828,     632511.375284974,     878871.667639863,
        1221188.18385947,     1696835.44079615,     2357745.16122063,
        3276076.23993788,     4552093.13615831,     6325112.86420063,
        8788715.78775012,
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_SOFT_EQ(expected_range, range.y);
    }
}

TEST_F(RangeGridCalculatorTest, mock)
{
    std::vector<real_type> dedx{0.5, 0.5, 0.5};
    this->build(1e-5, 10, make_span(dedx));

    // Linear interpolation will be used because there are fewer than 5 grid
    // points
    auto range = RangeGridCalculator(BC::geant)(dedx_grid);

    constexpr auto tol = SoftEqual<real_type>{}.rel();
    static double const expected_range[] = {4e-05, 0.02002, 20.00002};
    EXPECT_VEC_NEAR(expected_range, range.y, tol);
}

TEST_F(RangeGridCalculatorTest, sparse)
{
    std::vector<real_type> dedx{
        839.668353354807,
        430.530096695467,
        111.600220710967,
        22.6117194229536,
        10.6619173294951,
        11.0069268409596,
        11.3553238163283,
        11.3784262549454,
        11.378228777509,
        11.3782267757997,
        11.3782267557938,
        11.3782267555937,
        11.3782267555917,
    };
    this->build(1e-4, 1e8, make_span(dedx));

    {
        // Cubic splines can exhibit oscillations and instability when the
        // independent variable scale is large and the data is sparse. The
        // values can be off by orders of magnitude: in this case, some are
        // negative. Polynomial interpolation works better for this example,
        // but still has some negative values. Hopefully Geant4 always uses a
        // large enough grid for the energy loss table.
        RangeGridCalculator calc_range(BC::geant);
        EXPECT_THROW(calc_range(dedx_grid), RuntimeError);
    }
    {
        // Linear interpolation
        auto range = RangeGridCalculator()(dedx_grid);
        static double const expected_range[] = {
            2.3818927937551e-07,
            1.7075905920016e-06,
            3.9805502137378e-05,
            0.0016543439268271,
            0.058275147408879,
            0.88903109276619,
            8.9389616221692,
            88.116423057883,
            879.09383053631,
            8788.9372408309,
            87887.378370992,
            878871.79037494,
            8788715.9104847,
        };
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_VEC_SOFT_EQ(expected_range, range.y);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
