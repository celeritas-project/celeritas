//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ElementCdfCalculator.cc
//---------------------------------------------------------------------------//
#include "ElementCdfCalculator.hh"

#include <algorithm>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from elements in a material.
 */
ElementCdfCalculator::ElementCdfCalculator(SpanConstElement elements)
    : elements_(elements)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the CDF in place from the microscopic cross sections.
 */
void ElementCdfCalculator::operator()(XsTable& grids)
{
    CELER_EXPECT(grids.size() == elements_.size());
    CELER_EXPECT(std::all_of(grids.begin(), grids.end(), [](auto const& g) {
        return static_cast<bool>(g);
    }));

    // Outer loop over energy: the energy grid is the same for each element
    for (auto i : range(grids.front().y.size()))
    {
        // Calculate the CDF in place
        double accum{0};
        for (auto j : range(elements_.size()))
        {
            auto& value = grids[j].y[i];
            accum += value * elements_[j].fraction;
            value = accum;
        }
        if (accum > 0)
        {
            // Normalize the CDF
            double norm = 1 / accum;
            for (auto j : range(elements_.size() - 1))
            {
                grids[j].y[i] *= norm;
            }
            grids.back().y[i] = 1;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
