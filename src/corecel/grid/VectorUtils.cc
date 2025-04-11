//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/VectorUtils.cc
//---------------------------------------------------------------------------//
#include "VectorUtils.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

#include "Interpolator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<Interp YI>
std::vector<double> space_impl(double start, double stop, size_type n)
{
    std::vector<double> result(n);

    Interpolator<Interp::linear, YI, double> interp(
        {0.0, start}, {static_cast<double>(n - 1), stop});

    result.front() = start;
    for (auto i : range<size_type>(1, n - 1))
    {
        result[i] = interp(static_cast<double>(i));
    }
    // Manually set last point to avoid any differences due to roundoff
    result.back() = stop;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Return evenly spaced numbers over a given interval.
 */
std::vector<double> linspace(double start, double stop, size_type n)
{
    CELER_EXPECT(start < stop);
    CELER_EXPECT(n > 1);

    return space_impl<Interp::linear>(start, stop, n);
}

//---------------------------------------------------------------------------//
/*!
 * Return logarithmically spaced numbers over a given interval.
 *
 * Unlike numpy's logspace which assumes the start and stop are log-10 values
 * (unless given another argument), the start and stop are the *actual* first
 * and last values of the resulting vector.
 */
std::vector<double> logspace(double start, double stop, size_type n)
{
    CELER_EXPECT(0 < start);
    CELER_EXPECT(start < stop);
    CELER_EXPECT(n > 1);

    return space_impl<Interp::log>(start, stop, n);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
