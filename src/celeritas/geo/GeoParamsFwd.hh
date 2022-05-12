//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
#if CELERITAS_USE_VECGEOM
class VecgeomParams;
using GeoParams = VecgeomParams;
#else
class OrangeParams;
using GeoParams = OrangeParams;
#endif
//---------------------------------------------------------------------------//
} // namespace celeritas
