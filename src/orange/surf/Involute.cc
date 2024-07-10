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
/*
 * \param radius: radius of the circle of involute
 * \param a: displacement angle of the involute
 * \param sign: chirality of involute
 * \param tmin: minimum tangent angle
 * \param tmax: maximum tangent angle
 */

Involute::Involute(Real3 const& origin,
                   real_type radius,
                   real_type a,
                   real_type tmin,
                   real_type tmax)
    : origin_(origin)
    , r_b_(std::fabs(radius))
    , a_(a)
    , sign_(static_cast<Sign>(radius < 0))
    , tmin_(tmin)
    , tmax_(tmax)
{
    CELER_EXPECT(a > 0);
    CELER_EXPECT(tmax > 0);
    CELER_EXPECT(tmin > 0);
    CELER_EXPECT(tmax < 2 * pi + tmin);

    if (sign_)
    {
        a_ = pi - a;
    }
}
//---------------------------------------------------------------------------//
}  // namespace celeritas