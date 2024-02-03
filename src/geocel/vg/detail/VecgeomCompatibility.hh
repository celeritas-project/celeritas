//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomCompatibility.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Vector3D.h>

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a Vector3D from a length-3 span.
 */
template<class T>
CELER_FUNCTION inline auto to_vector(Span<T, 3> s)
    -> vecgeom::Vector3D<std::remove_cv_t<T>>
{
    return {s[0], s[1], s[2]};
}

//---------------------------------------------------------------------------//
/*!
 * Create a Vector3D from a length-3 array.
 */
template<class T>
CELER_FUNCTION inline auto to_vector(Array<T, 3> const& arr)
    -> vecgeom::Vector3D<T>
{
    return to_vector(celeritas::make_span<T, 3>(arr));
}

//---------------------------------------------------------------------------//
/*!
 * Create a length-3 array from a VecGeom vector.
 */
template<class T>
CELER_FUNCTION inline auto to_array(vecgeom::Vector3D<T> const& arr)
    -> Array<T, 3>
{
    return {arr[0], arr[1], arr[2]};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
