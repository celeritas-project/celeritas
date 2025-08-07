//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Whether a track is entering the surface defined by the given normal.
 *
 * The surface normal convention used in Celeritas optical physics is that
 * the normal direction points opposite the incident track direction. This
 * function makes checks for this condition explicit in the code.
 */
inline CELER_FUNCTION bool
is_entering_surface(Real3 const& normal, Real3 const& dir)
{
    return dot_product(normal, dir) < 0;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
