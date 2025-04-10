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
 * Contruct with the number of bins and domain.
 */
Histogram::Histogram(size_type num_bins, Dbl2 domain)
    : offset_{domain[0]}
    , inv_width_(1 / (domain[1] - domain[0]))
    , counts_(num_bins)
{
    CELER_EXPECT(num_bins > 0);
    CELER_EXPECT(domain[0] < domain[1]);
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
/*!
 * Calculate the total number of samples within the domain.
 */
size_type Histogram::calc_valid_samples() const
{
    return std::accumulate(counts_.begin(), counts_.end(), size_type{0});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
