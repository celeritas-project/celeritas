//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseCdfFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Given a sampled CDF value, find the corresponding grid value.
 *
 * \tparam G Grid, e.g. \c UniformGrid or \c NonuniformGrid
 * \tparam C Calculate the CDF at a given grid index
 *
 * Both the input grid and the CDF must be monotonically increasing. The
 * sampled CDF value must be in range.
 */
template<class G, class C>
class InverseCdfFinder
{
  public:
    // Construct from grid and CDF calculator
    inline CELER_FUNCTION InverseCdfFinder(G&& grid, C&& calc_cdf);

    // Find and interpolate the grid value corresponding to the given CDF
    inline CELER_FUNCTION real_type operator()(real_type cdf) const;

  private:
    G grid_;
    C calc_cdf_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid and CDF calculator.
 */
template<class G, class C>
CELER_FUNCTION InverseCdfFinder<G, C>::InverseCdfFinder(G&& grid, C&& calc_cdf)
    : grid_(celeritas::forward<G>(grid))
    , calc_cdf_(celeritas::forward<C>(calc_cdf))
{
    CELER_EXPECT(grid_.size() >= 2);
    CELER_EXPECT(calc_cdf_[0] == 0
                 && soft_equal(calc_cdf_[grid_.size() - 1], real_type(1)));
}

//---------------------------------------------------------------------------//
/*!
 * Find and interpolate the grid value corresponding to the given CDF.
 */
template<class G, class C>
CELER_FUNCTION real_type InverseCdfFinder<G, C>::operator()(real_type cdf) const
{
    CELER_EXPECT(cdf >= 0 && cdf < 1);

    // Find the grid index of the sampled CDF value
    Range indices(grid_.size() - 1);
    auto iter = celeritas::lower_bound(
        indices.begin(), indices.end(), cdf, [this](size_type i, real_type c) {
            return calc_cdf_[i] < c;
        });
    if (calc_cdf_[*iter] != cdf)
    {
        --iter;
    }
    size_type i = iter - indices.begin();

    // Calculate the grid value corresponding to the sampled CDF value
    return LinearInterpolator<real_type>{
        {calc_cdf_[i], grid_[i]}, {calc_cdf_[i + 1], grid_[i + 1]}}(cdf);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
