//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef geometry_Types_hh
#define geometry_Types_hh

#include "base/OpaqueId.hh"

namespace celeritas
{
class Geometry;
//---------------------------------------------------------------------------//
using VolumeId = OpaqueId<Geometry, unsigned int>;

enum class Boundary : bool
{
    outside = false,
    inside  = true
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_Types_hh
