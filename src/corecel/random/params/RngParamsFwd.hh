//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/RngParamsFwd.hh
//! \brief Forward-declare RngParams alias.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

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
