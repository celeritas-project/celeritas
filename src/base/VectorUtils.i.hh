//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VectorUtils.i.hh
//---------------------------------------------------------------------------//

#include "Assert.hh"
#include "base/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return evenly spaced numbers over a specific interval
 */
template<class T>
std::vector<real_type> linspace(T start, T stop, size_type n)
{
    CELER_EXPECT(n > 1);
    std::vector<real_type> result(n);

    // Convert input values to real_type
    real_type start_c = start;
    real_type stop_c  = stop;

    // Build vector of evenly spaced numbers
    real_type delta = (stop_c - start_c) / (n - 1);
    for (auto i : range(n - 1))
    {
        result[i] = start_c + delta * i;
    }
    // Manually add last point to avoid any differences due to roundoff
    result[n - 1] = stop_c;
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
