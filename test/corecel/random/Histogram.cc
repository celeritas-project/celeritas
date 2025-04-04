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
Histogram::Histogram(size_type num_bins, Real2 range)
    : range_(range), width_((range[1] - range[0]) / num_bins), counts_(num_bins)
{
    CELER_EXPECT(num_bins > 0);
    CELER_EXPECT(range[0] < range[1]);
}

//---------------------------------------------------------------------------//
/*!
 * Update the histogram with a value.
 *
 * Values outside of \c range will be ignored.
 */
void Histogram::operator()(real_type value)
{
    if (value < range_[0] || value >= range_[1])
    {
        return;
    }
    auto index = static_cast<size_type>((value - range_[0]) / width_);
    CELER_ASSERT(index < counts_.size());
    ++counts_[index];
}

//---------------------------------------------------------------------------//
/*!
 * Get the result as a probability desnity.
 */
auto Histogram::density() const -> VecReal
{
    VecReal result;
    result.reserve(counts_.size());

    auto norm = real_type(1)
                / std::accumulate(counts_.begin(), counts_.end(), size_type(0));

    for (auto count : counts_)
    {
        result.push_back(count * norm);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
