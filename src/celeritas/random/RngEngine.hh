//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

// Alias core RNG type using on compile-time RNG selection
#if (CELERITAS_RNG == CELERITAS_RNG_CURAND) \
    || (CELERITAS_RNG == CELERITAS_RNG_HIPRAND)
#    include "CuHipRngEngine.hh"
namespace celeritas
{
using RngEngine = CuHipRngEngine;
}
#elif (CELERITAS_RNG == CELERITAS_RNG_XORWOW)
#    include "XorwowRngEngine.hh"
namespace celeritas
{
using RngEngine = XorwowRngEngine;
}
#endif
