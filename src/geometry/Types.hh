//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"

namespace celeritas
{
class Geometry;
//---------------------------------------------------------------------------//
using VolumeId = OpaqueId<Geometry, unsigned int>;

/*!
 * Whether the particle is inside or outside the problem space.
 *
 * The particle tracking should terminate when the boundary state is "outside".
 */
enum class Boundary : bool
{
    outside = false,
    inside  = true
};

//---------------------------------------------------------------------------//
} // namespace celeritas
