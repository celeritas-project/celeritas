//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file curand.nocuda.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Placeholder for CUDA random state to allow compiling
struct MockCurandState
{
};

//---------------------------------------------------------------------------//
//!@{
//! CUDA random functions.
void         curand_init(unsigned long long seed,
                         unsigned long long sequence,
                         unsigned long long offset,
                         MockCurandState*   state);
unsigned int curand(MockCurandState* state);
float        curand_uniform(MockCurandState* state);
double       curand_uniform_double(MockCurandState* state);
//!@}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
