//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoTrackView.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_VECGEOM
#    include "celeritas/ext/VecgeomTrackView.hh"
#else
#    include "orange/OrangeTrackView.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_VECGEOM
using GeoTrackView = VecgeomTrackView;
#else
using GeoTrackView = OrangeTrackView;
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
