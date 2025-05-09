//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ElementCdfCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/ElementCdfCalculator.hh"

#include <vector>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class ElementCdfCalculatorTest : public Test
{
  protected:
    using SpanConstElement = ElementCdfCalculator::SpanConstElement;
    using VecElement = std::vector<MatElementComponent>;
    using VecDbl = std::vector<double>;
    using VecVecDbl = std::vector<VecDbl>;
    using XsTable = ElementCdfCalculator::XsTable;

    //! Create an element vector from the fractional composition
    void make_elements(VecDbl fraction)
    {
        elements = VecElement(fraction.size());
        for (auto i : range(elements.size()))
        {
            elements[i].element = ElementId(i);
            elements[i].fraction = fraction[i];
        }
    }

    //! Create a micro xs table indexed as [element][energy]
    void make_grids(VecVecDbl xs)
    {
        grids = XsTable(xs.size());
        for (auto i : range(grids.size()))
        {
            grids[i].lower.x = {0, 1e3};
            grids[i].lower.y = xs[i];
        }
    }

    //! Get the CDF values indexed as [energy][element]
    VecVecDbl get_cdf(XsTable const& cdf)
    {
        auto grid_size = cdf.front().lower.y.size();
        VecVecDbl result(grid_size);
        for (auto i : range(grid_size))
        {
            result[i].resize(cdf.size());
            for (auto j : range(cdf.size()))
            {
                result[i][j] = cdf[j].lower.y[i];
            }
        }
        return result;
    }

    VecElement elements;
    XsTable grids;
};

TEST_F(ElementCdfCalculatorTest, calc_cdf)
{
    {
        make_elements({0.25, 0.25, 0.25, 0.25});
        make_grids({
            {1, 1, 1, 1},
            {2, 2, 2, 2},
            {3, 3, 3, 3},
            {4, 4, 4, 4},
        });

        ElementCdfCalculator(make_span(elements))(grids);
        static std::vector<double> const expected_cdf[] = {
            {0.1, 0.3, 0.6, 1},
            {0.1, 0.3, 0.6, 1},
            {0.1, 0.3, 0.6, 1},
            {0.1, 0.3, 0.6, 1},
        };
        EXPECT_VEC_SOFT_EQ(expected_cdf, get_cdf(grids));
    }
    {
        make_elements({0.25, 0.125, 0.5, 0.125});
        make_grids({
            {1, 10, 100, 1000},
            {1, 10, 100, 1000},
            {1, 10, 100, 1000},
            {1, 10, 100, 1000},
        });

        ElementCdfCalculator(make_span(elements))(grids);
        static std::vector<double> const expected_cdf[] = {
            {0.25, 0.375, 0.875, 1},
            {0.25, 0.375, 0.875, 1},
            {0.25, 0.375, 0.875, 1},
            {0.25, 0.375, 0.875, 1},
        };
        EXPECT_VEC_SOFT_EQ(expected_cdf, get_cdf(grids));
    }
    {
        make_elements({1});
        make_grids({{0.1, 1, 10, 100, 1000}});

        ElementCdfCalculator(make_span(elements))(grids);
        static std::vector<double> const expected_cdf[]
            = {{1}, {1}, {1}, {1}, {1}};
        EXPECT_VEC_SOFT_EQ(expected_cdf, get_cdf(grids));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
