//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoParamsFwd.hh
//! \brief Forward-declare configure-time geometry implementation
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
class VecgeomParams;
using GeoParams = VecgeomParams;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
class OrangeParams;
using GeoParams = OrangeParams;
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
