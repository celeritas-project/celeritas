//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoParams.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_VECGEOM
#    include "celeritas/ext/VecgeomParams.hh"
#else
#    include "orange/OrangeParams.hh"
#endif

#include "GeoParamsFwd.hh"
