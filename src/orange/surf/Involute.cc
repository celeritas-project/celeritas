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
Involute::Involute(Real3 const& origin,
                   real_type radius,
                   real_type a,
                   real_type sign,
                   real_type tmin,
                   real_type tmax)
    : origin_(origin), r_b_(radius), a_(a), sign_(sign), tmin_(tmin), tmax_(tmax)
{
    CELER_EXPECT(radius > 0);
    CELER_EXPECT(a > 0);
    CELER_EXPECT(std::fabs(tmax) < 2 * pi + std::fabs(tmin));
}
//---------------------------------------------------------------------------//
}  // namespace celeritas