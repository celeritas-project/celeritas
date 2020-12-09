//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file curand.nocuda.hh
//! \brief Support compiling/linking but raise runtime errors.
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    error "This file should only be included when CUDA is unavailable."
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Placeholder for CUDA random state to allow compiling
struct MockCurandState
{
};

using curandState_t = MockCurandState;

//---------------------------------------------------------------------------//
//@{
//! CUDA random functions.
void         curand_init(unsigned long long seed,
                         unsigned long long sequence,
                         unsigned long long offset,
                         curandState_t*     state);
unsigned int curand(curandState_t* state);
float        curand_uniform(curandState_t* state);
double       curand_uniform_double(curandState_t* state);
//@}

//---------------------------------------------------------------------------//
} // namespace celeritas
