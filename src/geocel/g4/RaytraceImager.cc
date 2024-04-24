//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/RaytraceImager.cc
//---------------------------------------------------------------------------//
#include "RaytraceImager.hh"

#include "geocel/rasterize/RaytraceImager.nocuda.t.hh"
#include "geocel/rasterize/RaytraceImager.t.hh"

#include "GeantGeoData.hh"
#include "GeantGeoParams.hh"
#include "GeantGeoTrackView.hh"
#include "GeantGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class RaytraceImager<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
