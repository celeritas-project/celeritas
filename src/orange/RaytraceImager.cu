//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/RaytraceImager.cu
//---------------------------------------------------------------------------//
#include "RaytraceImager.hh"

#include "geocel/rasterize/RaytraceImager.cuda.t.hh"

#include "OrangeData.hh"
#include "OrangeGeoTraits.hh"
#include "OrangeParams.hh"
#include "OrangeTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class RaytraceImager<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
