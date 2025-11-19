//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomCompatibility.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Vector3D.h>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "geocel/vg/VecgeomTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a Vector3D from a length-3 span.
 */
template<class T>
inline CELER_FUNCTION auto to_vector(Span<T, 3> s)
    -> VgVector3<std::remove_cv_t<T>, MemSpace::native>
{
    return {s[0], s[1], s[2]};
}

//---------------------------------------------------------------------------//
/*!
 * Create a Vector3D from a length-3 array.
 */
template<class T>
inline CELER_FUNCTION auto to_vector(Array<T, 3> const& arr)
    -> VgVector3<T, MemSpace::native>
{
    return to_vector(celeritas::make_span<T, 3>(arr));
}

//---------------------------------------------------------------------------//
/*!
 * Create a length-3 array from a VecGeom vector.
 */
template<class T>
inline CELER_FUNCTION auto to_array(vecgeom::Vector3D<T> const& vgv)
    -> Array<T, 3>
{
    return {vgv[0], vgv[1], vgv[2]};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
