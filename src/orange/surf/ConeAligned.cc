//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/ConeAligned.cc
//---------------------------------------------------------------------------//
#include "ConeAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a new origin using the radius of another cone.
 */
template<Axis T>
ConeAligned<T>::ConeAligned(Real3 const& origin, ConeAligned const& other)
    : origin_{origin}, tsq_{other.tsq_}
{
}

//---------------------------------------------------------------------------//

template class ConeAligned<Axis::x>;
template class ConeAligned<Axis::y>;
template class ConeAligned<Axis::z>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
