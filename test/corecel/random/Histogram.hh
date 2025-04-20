//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Histogram.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

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

  private:
    double offset_;
    double inv_width_;
    VecCount counts_;
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
    if (frac < 0.0 || frac > 1.0)
    {
        return;
    }
    auto index = static_cast<size_type>(frac * counts_.size());
    if (frac == 1.0)
    {
        CELER_ASSERT(index == counts_.size());
        --index;
    }
    CELER_ASSERT(index < counts_.size());
    ++counts_[index];
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
