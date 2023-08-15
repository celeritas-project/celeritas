//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylAligned.cc
//---------------------------------------------------------------------------//
#include "CylAligned.hh"

#include "CylCentered.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from the square of the radius and the origin.
 *
 * This is used for surface simplification.
 */
template<Axis T>
CylAligned<T> CylAligned<T>::from_radius_sq(Real3 const& origin, real_type rsq)
{
    CELER_EXPECT(rsq > 0);

    CylAligned<T> result;
    result.origin_u_ = origin[to_int(U)];
    result.origin_v_ = origin[to_int(V)];
    result.radius_sq_ = rsq;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a centered axis-aligned cylinder.
 */
template<Axis T>
CylAligned<T>::CylAligned(CylCentered<T> const& other) noexcept
    : origin_u_{0}, origin_v_{0}, radius_sq_{other.radius_sq()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the origin as a 3-vector.
 */
template<Axis T>
Real3 CylAligned<T>::calc_origin() const
{
    Real3 result{0, 0, 0};
    result[to_int(U)] = this->origin_u();
    result[to_int(V)] = this->origin_v();
    return result;
}

//---------------------------------------------------------------------------//

template class CylAligned<Axis::x>;
template class CylAligned<Axis::y>;
template class CylAligned<Axis::z>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
