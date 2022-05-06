//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

// Alias core RNG type using on compile-time RNG selection
#if (CELERITAS_RNG == CELERITAS_RNG_CURAND) \
    || (CELERITAS_RNG == CELERITAS_RNG_HIPRAND)
#    include "CuHipRngParams.hh"
namespace celeritas
{
using RngParams = CuHipRngParams;
}
#elif (CELERITAS_RNG == CELERITAS_RNG_XORWOW)
#    include "XorwowRngParams.hh"
namespace celeritas
{
using RngParams = XorwowRngParams;
}
#endif
