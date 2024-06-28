//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.cc
//---------------------------------------------------------------------------//
#include "Involute.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
Involute Involute::at0origin(Real3 const& origin,
                             real_type radius,
                             real_type a,
                             real_type sign,
                             real_type tmin,
                             real_type tmax)
{
    CELER_EXPECT(radius > 0);
    CELER_EXPECT(a > 0);
    Involute results;
    results.origin_ = origin;
    results.r_b_ = radius;
    results.a_ = a;
    results.sign_ = sign;
    results.tmin_ = tmin;
    results.tmax_ = tmax;
    return results;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas