//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/CoreGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "celeritas/ext/VecgeomTestBase.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeTestBase.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
#    include "celeritas/ext/GeantGeoTestBase.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
using CoreGeoTestBase = VecgeomTestBase;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
using CoreGeoTestBase = OrangeTestBase;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
using CoreGeoTestBase = GeantGeoTestBase;
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
