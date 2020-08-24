//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStatePointers.cuh
//---------------------------------------------------------------------------//
#pragma once

#include <curand_kernel.h>
#include "base/Span.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! State data for a CUDA RNG
using RngState = curandState_t;

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
