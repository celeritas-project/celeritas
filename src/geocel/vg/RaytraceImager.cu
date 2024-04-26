//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/RaytraceImager.cu
//---------------------------------------------------------------------------//
#include "RaytraceImager.hh"

#include "geocel/rasterize/RaytraceImager.cuda.t.hh"

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
