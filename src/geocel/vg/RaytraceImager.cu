//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/RaytraceImager.cu
//---------------------------------------------------------------------------//
#include "RaytraceImager.hh"

#include "geocel/rasterize/RaytraceImager.device.t.hh"

#include "VecgeomData.hh"
#include "VecgeomGeoTraits.hh"
#include "VecgeomParams.hh"
#include "VecgeomTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class RaytraceImager<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
