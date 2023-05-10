//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoData.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "celeritas/ext/VecgeomData.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeData.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
template<Ownership W, MemSpace M>
using GeoParamsData = VecgeomParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = VecgeomStateData<W, M>;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
template<Ownership W, MemSpace M>
using GeoParamsData = OrangeParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = OrangeStateData<W, M>;
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
