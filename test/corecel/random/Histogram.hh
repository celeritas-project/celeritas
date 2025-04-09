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

    // Get the histogram
    VecCount const& counts() const { return counts_; }

    // Get the result as a probability density
    VecDbl calc_density() const;

  private:
    Dbl2 range_;
    double width_;
    VecCount counts_;
    size_type total_counts_{0};
};

//---------------------------------------------------------------------------//
/*!
 * Update the histogram with a value.
 *
 * Values outside of \c range are allowable and will show as a deficit in the
 * resulting tally.
 */
void Histogram::operator()(double value)
{
    ++total_counts_;
    if (value < range_[0] || value >= range_[1])
    {
        return;
    }
    auto index = static_cast<size_type>((value - range_[0]) / width_);
    CELER_ASSERT(index < counts_.size());
    ++counts_[index];
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
