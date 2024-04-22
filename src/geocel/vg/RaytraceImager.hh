//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/RaytraceImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/rasterize/RaytraceImager.hh"

#include "VecgeomGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

extern template class RaytraceImager<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
