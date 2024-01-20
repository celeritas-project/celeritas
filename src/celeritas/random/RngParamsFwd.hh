//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngParamsFwd.hh
//! \brief Forward-declare RngParams alias.
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

namespace celeritas
{
// Alias core RNG type using on compile-time RNG selection
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_CURAND) \
    || (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_HIPRAND)
class CuHipRngParams;
using RngParams = CuHipRngParams;
#elif (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
class XorwowRngParams;
using RngParams = XorwowRngParams;
#endif
}  // namespace celeritas
