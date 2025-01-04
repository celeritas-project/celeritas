//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/RaytraceImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/rasterize/RaytraceImager.hh"

#include "VecgeomGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

extern template class RaytraceImager<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
