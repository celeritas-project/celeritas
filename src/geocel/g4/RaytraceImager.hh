//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/RaytraceImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/rasterize/RaytraceImager.hh"

#include "GeantGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

extern template class RaytraceImager<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
