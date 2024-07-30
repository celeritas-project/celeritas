//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.cc
//---------------------------------------------------------------------------//
#include "Involute.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct involute from parameters.
 *
 * \param origin origin of the involute
 * \param radius radius of the circle of involute
 * \param displacement displacement angle of the involute
 * \param sign chirality of involute
 * \param tmin minimum tangent angle
 * \param tmax maximum tangent angle
 */

Involute::Involute(Real2 const& origin,
                   real_type radius,
                   real_type displacement,
                   Sign sign,
                   real_type tmin,
                   real_type tmax)
    : origin_(origin), r_b_(radius), a_(displacement), tmin_(tmin), tmax_(tmax)
{
    CELER_EXPECT(r_b_ >= 0);
    CELER_EXPECT(a_ >= 0 && a_ <= 2 * constants::pi);
    CELER_EXPECT(tmin_ >= 0);
    CELER_EXPECT(tmax_ > tmin_ && tmax_ < 2 * constants::pi + tmin_);

    if (sign)
    {
        a_ = constants::pi - a_;
        r_b_ = -r_b_;
    }

    CELER_ENSURE(this->r_b() == radius);
    CELER_ENSURE(this->sign() == sign);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
