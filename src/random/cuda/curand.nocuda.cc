//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file curand.nocuda.cc
//---------------------------------------------------------------------------//
#include "curand.nocuda.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void curand_init(unsigned long long,
                 unsigned long long,
                 unsigned long long,
                 curandState_t*)
{
    CELER_ASSERT_UNREACHABLE();
}

unsigned int curand(curandState_t*)
{
    CELER_ASSERT_UNREACHABLE();
}

float curand_uniform(curandState_t*)
{
    CELER_ASSERT_UNREACHABLE();
}

double curand_uniform_double(curandState_t*)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
