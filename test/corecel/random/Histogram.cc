//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Histogram.cc
//---------------------------------------------------------------------------//
#include "Histogram.hh"

#include <numeric>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Contruct with the number of bins and range.
 */
Histogram::Histogram(size_type num_bins, Dbl2 range)
    : offset_{range[0]}
    , inv_width_(1 / (range[1] - range[0]))
    , counts_(num_bins)
{
    CELER_EXPECT(num_bins > 0);
    CELER_EXPECT(range[0] < range[1]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the result as a probability desnity.
 */
auto Histogram::calc_density() const -> VecDbl
{
    VecDbl result;
    result.reserve(counts_.size());

    auto norm = 1 / static_cast<double>(total_counts_);
    for (auto count : counts_)
    {
        result.push_back(count * norm);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
