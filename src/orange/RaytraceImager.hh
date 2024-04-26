//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/RaytraceImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/rasterize/RaytraceImager.hh"

#include "OrangeGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

extern template class RaytraceImager<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
