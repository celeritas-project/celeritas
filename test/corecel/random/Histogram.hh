//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Histogram.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/grid/GridTypes.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the histogram of a set of values.
 *
 * This uses double precision since values are being accumulated and tallied.
 */
class Histogram
{
  public:
    //!@{
    //! \name Type aliases
    using Dbl2 = Array<double, 2>;
    using VecCount = std::vector<size_type>;
    using VecDbl = std::vector<double>;
    //!@}

  public:
    // Construct with number of bins and range
    Histogram(size_type num_bins, Dbl2 range);

    // Update the histogram with a value
    inline void operator()(double value);

    // Update the histogram with a vector of values
    inline void operator()(VecDbl const& values);

    // Get the histogram
    VecCount const& counts() const { return counts_; }

    // Get the result as a probability density
    VecDbl calc_density() const;

    // Get the overflow and overflow counts
    size_type underflow() const { return out_of_range_[Bound::lo]; }
    size_type overflow() const { return out_of_range_[Bound::hi]; }

    // Get the minimum and maximum values encountered
    double min() const { return extrema_[Bound::lo]; }
    double max() const { return extrema_[Bound::hi]; }

  private:
    using Limits = std::numeric_limits<double>;

    double offset_;
    double inv_width_;
    VecCount counts_;

    // Keep track of underflow and overflow
    EnumArray<Bound, size_type> out_of_range_{};
    EnumArray<Bound, double> extrema_{Limits::infinity(), -Limits::infinity()};
};

//---------------------------------------------------------------------------//
/*!
 * Update the histogram with a value.
 *
 * Values outside of \c range are ignored. All bins are half-open except for
 * the rightmost bin, which will include values equal to the upper edge.
 */
void Histogram::operator()(double value)
{
    double frac = (value - offset_) * inv_width_;
    if (frac < 0.0)
    {
        ++out_of_range_[Bound::lo];
        extrema_[Bound::lo] = std::min(extrema_[Bound::lo], value);
    }
    else if (frac < 1.0)
    {
        auto index = static_cast<size_type>(frac * counts_.size());
        CELER_ASSERT(index < counts_.size());
        ++counts_[index];
    }
    else if (frac == 1.0)
    {
        ++counts_.back();
    }
    else
    {
        ++out_of_range_[Bound::hi];
        extrema_[Bound::hi] = std::max(extrema_[Bound::hi], value);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update the histogram with a vector of values.
 */
void Histogram::operator()(VecDbl const& values)
{
    for (auto v : values)
    {
        (*this)(v);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
