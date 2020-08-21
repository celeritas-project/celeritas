//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStatePointers.cuh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <curand_kernel.h>
#endif
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! State data for a CUDA RNG
#if CELERITAS_USE_CUDA
using RngState = curandState_t;
#else
using RngState = int;
#endif

//---------------------------------------------------------------------------//
//! Initializer for an RNG
struct RngSeed
{
    using value_type = unsigned long long;
    value_type seed;
};

//---------------------------------------------------------------------------//
/*!
 * Device pointers to a vector of CUDA random number generator states.
 */
struct RngStatePointers
{
    span<RngState> rng;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
