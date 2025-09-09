//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/RaytraceImager.cc
//---------------------------------------------------------------------------//
#include "RaytraceImager.hh"

#include "geocel/rasterize/RaytraceImager.nocuda.t.hh"
#include "geocel/rasterize/RaytraceImager.t.hh"

#include "GeantGeoData.hh"
#include "GeantGeoTrackView.hh"
#include "GeantGeoTraits.hh"
#include "../GeantGeoParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class RaytraceImager<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
