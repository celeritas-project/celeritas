//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/VectorUtils.cc
//---------------------------------------------------------------------------//
#include "VectorUtils.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return evenly spaced numbers over a given interval.
 */
std::vector<real_type> linspace(real_type start, real_type stop, size_type n)
{
    CELER_EXPECT(n > 1);
    std::vector<real_type> result(n);

    // Build vector of evenly spaced numbers
    real_type delta = (stop - start) / (n - 1);
    for (auto i : range(n - 1))
    {
        result[i] = start + delta * i;
    }
    // Manually add last point to avoid any differences due to roundoff
    result[n - 1] = stop;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
