//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/RngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

// IWYU pragma: begin_exports
// Alias core RNG type using on compile-time RNG selection
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_CURAND) \
    || (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_HIPRAND)
#    include "CuHipRngEngine.hh"
namespace celeritas
{
using RngEngine = CuHipRngEngine;
}
#elif (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
#    include "XorwowRngEngine.hh"
namespace celeritas
{
using RngEngine = XorwowRngEngine;
}
#endif
// IWYU pragma: end_exports
