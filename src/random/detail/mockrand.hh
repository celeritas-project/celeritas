//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file mockrand.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Placeholder for CUDA random state to allow compiling
struct MockRandState
{
};

//---------------------------------------------------------------------------//
//!@{
//! Random functions.
void         mockrand_init(unsigned long long seed,
                           unsigned long long sequence,
                           unsigned long long offset,
                           MockRandState*     state);
unsigned int mockrand(MockRandState* state);
float        mockrand_uniform(MockRandState* state);
double       mockrand_uniform_double(MockRandState* state);
//!@}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
