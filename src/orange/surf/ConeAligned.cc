//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/ConeAligned.cc
//---------------------------------------------------------------------------//
#include "ConeAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from origin and tangent of the angle of its opening.
 */
template<Axis T>
ConeAligned<T>::ConeAligned(Real3 const& origin, real_type tangent)
    : origin_{origin}, tsq_{ipow<2>(tangent)}
{
    CELER_EXPECT(tsq_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with square of tangent for simplification.
 */
template<Axis T>
ConeAligned<T>
ConeAligned<T>::from_tangent_sq(Real3 const& origin, real_type tsq)
{
    CELER_EXPECT(tsq > 0);
    ConeAligned result;
    result.origin_ = origin;
    result.tsq_ = tsq;
    return result;
}

//---------------------------------------------------------------------------//

template class ConeAligned<Axis::x>;
template class ConeAligned<Axis::y>;
template class ConeAligned<Axis::z>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
