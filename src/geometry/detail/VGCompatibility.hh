//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGCompatibility.h
//---------------------------------------------------------------------------//
#ifndef geometry_detail_Compatibility_h
#define geometry_detail_Compatibility_h

#include <VecGeom/base/Vector3D.h>

#include "base/Macros.hh"
#include "base/Array.hh"
#include "base/Span.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Copy a length-3 span into a Vector3D
 */
template<class T>
CELER_FUNCTION inline auto to_vector(span<T, 3> s)
    -> vecgeom::Vector3D<std::remove_cv_t<T>>
{
    return {s[0], s[1], s[2]};
}

//---------------------------------------------------------------------------//
// Copy a length-3 array into a Vector3D
template<class T>
CELER_FUNCTION inline auto to_vector(const array<T, 3>& arr)
    -> vecgeom::Vector3D<T>
{
    return to_vector(celeritas::make_span<T, 3>(arr));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#endif // geometry_detail_VGCompatibility_h
