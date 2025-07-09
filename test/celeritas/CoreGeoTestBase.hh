//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/CoreGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "geocel/vg/VecgeomTestBase.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeTestBase.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
#    include "geocel/g4/GeantGeoTestBase.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Testing class for the core geometry
class CoreGeoTestBase : public
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
                        VecgeomTestBase
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
                        OrangeTestBase
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
                        GeantGeoTestBase
#endif
{
  public:
    using SPConstCoreGeo = SPConstGeo;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
