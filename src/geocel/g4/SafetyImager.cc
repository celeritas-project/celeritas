//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/SafetyImager.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/SafetyImager.t.hh"

#include "GeantGeoData.hh"
#include "GeantGeoTrackView.hh"
#include "GeantGeoTraits.hh"
#include "../GeantGeoParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class SafetyImager<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
