//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

// IWYU pragma: begin_exports
// Alias core RNG type using on compile-time RNG selection
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_CURAND) \
    || (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_HIPRAND)
#    include "CuHipRngData.hh"
namespace celeritas
{
template<Ownership W, MemSpace M>
using RngParamsData = CuHipRngParamsData<W, M>;
template<Ownership W, MemSpace M>
using RngStateData = CuHipRngStateData<W, M>;
}  // namespace celeritas
#elif (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW)
#    include "XorwowRngData.hh"
namespace celeritas
{
template<Ownership W, MemSpace M>
using RngParamsData = XorwowRngParamsData<W, M>;
template<Ownership W, MemSpace M>
using RngStateData = XorwowRngStateData<W, M>;
}  // namespace celeritas
#endif
// IWYU pragma: end_exports
