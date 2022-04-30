//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_VECGEOM
#    include "vecgeom/VecgeomParams.hh"
#else
#    include "orange/OrangeParams.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_VECGEOM
using GeoParams = VecgeomParams;
#else
using GeoParams = OrangeParams;
#endif
//---------------------------------------------------------------------------//
} // namespace celeritas
