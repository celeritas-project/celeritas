//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseCdfFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Given a sampled CDF value, find the corresponding grid value.
 *
 * The input grid should be monotonically increasing, and the given CDF value
 * must be in range.
 */
template<class GridT, class CalcCdf>
class InverseCdfFinder
{
  public:
    // Construct from grid and CDF calculator
    inline CELER_FUNCTION
    InverseCdfFinder(GridT const& grid, CalcCdf&& calc_cdf);

    // Find the grid value corresponding to the given CDF
    inline CELER_FUNCTION real_type operator()(real_type cdf) const;

  private:
    GridT const& grid_;
    CalcCdf calc_cdf_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid and CDF calculator.
 */
template<class GridT, class CalcCdf>
CELER_FUNCTION
InverseCdfFinder<GridT, CalcCdf>::InverseCdfFinder(GridT const& grid,
                                                   CalcCdf&& calc_cdf)
    : grid_(grid), calc_cdf_(std::move(calc_cdf))
{
}

//---------------------------------------------------------------------------//
/*!
 * Find the grid value corresponding to the given CDF.
 */
template<class GridT, class CalcCdf>
CELER_FUNCTION real_type
InverseCdfFinder<GridT, CalcCdf>::operator()(real_type cdf) const
{
    CELER_EXPECT(cdf >= 0 && cdf < 1);

    // Find the grid index of the sampled CDF value
    Range indices(grid_.size());
    auto iter = celeritas::lower_bound(
        indices.begin(), indices.end(), cdf, [this](size_type i, real_type c) {
            return calc_cdf_(i) < c;
        });
    CELER_ASSERT(iter != indices.end());
    size_type i = iter - indices.begin();
    CELER_ASSERT(i > 0);

    // Calculate the grid value corresponding to the sampled CDF value
    return LinearInterpolator<real_type>{{calc_cdf_(i - 1), grid_[i - 1]},
                                         {calc_cdf_(i), grid_[i]}}(cdf);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
