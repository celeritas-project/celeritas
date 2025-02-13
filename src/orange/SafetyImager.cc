//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/SafetyImager.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/SafetyImager.t.hh"

#include "OrangeData.hh"
#include "OrangeGeoTraits.hh"
#include "OrangeParams.hh"
#include "OrangeTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class SafetyImager<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
