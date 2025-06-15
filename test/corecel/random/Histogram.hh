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
 * Values outside of the input \c domain are saved to the underflow and
 * overflow bins. All bins are half-open except for the rightmost bin, which
 * will include values equal to the upper domain boundary.
 *
 * To test that all samples are within the domain:
 * \code
     EXPECT_EQ(0, hist.underflow())
       << "Encountered values as low as " << hist.min();
     EXPECT_EQ(0, hist.overflow())
       << "Encountered values as high as " << hist.max();
 * \endcode
 *
 * \note This class uses double precision since values are being accumulated
 * and tallied and only on host.
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
    // Construct with number of bins and domain
    Histogram(size_type num_bins, Dbl2 domain);

    // Update the histogram with a value
    inline void operator()(double value);

    // Update the histogram with a vector of values
    inline void operator()(VecDbl const& values);

    //! Get the histogram
    VecCount const& counts() const { return counts_; }

    // Get the result as a probability density
    VecDbl calc_density() const;

    //! Get the number of samples below the lower bound
    size_type underflow() const { return out_of_range_[Bound::lo]; }
    //! Get the number of samples above the upper bound
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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update the histogram with a value.
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
