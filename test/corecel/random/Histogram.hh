//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Histogram.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the histogram of a set of values.
 */
class Histogram
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    using VecCount = std::vector<size_type>;
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct with number of bins and range
    Histogram(size_type num_bins, Real2 range);

    // Update the histogram with a value
    void operator()(real_type value);

    // Get the histogram
    VecCount const& counts() const { return counts_; }

    // Get the result as a probability density
    VecReal density() const;

  private:
    Real2 range_;
    real_type width_;
    VecCount counts_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
