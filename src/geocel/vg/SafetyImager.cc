//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/SafetyImager.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/SafetyImager.t.hh"

#include "VecgeomData.hh"
#include "VecgeomGeoTraits.hh"
#include "VecgeomParams.hh"
#include "VecgeomTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class SafetyImager<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
