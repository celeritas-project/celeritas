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
 * Promote implicitly from a centered axis-aligned cylinder.
 */
template<Axis T>
CylAligned<T>::CylAligned(CylCentered<T> const& other)
    : origin_u_{0}, origin_v_{0}, radius_sq_{other.radius_sq()}
{
}

//---------------------------------------------------------------------------//

template class CylAligned<Axis::x>;
template class CylAligned<Axis::y>;
template class CylAligned<Axis::z>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
