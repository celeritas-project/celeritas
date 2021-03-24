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
namespace detail
{
//---------------------------------------------------------------------------//
void curand_init(unsigned long long,
                 unsigned long long,
                 unsigned long long,
                 MockCurandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

unsigned int curand(MockCurandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

float curand_uniform(MockCurandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

double curand_uniform_double(MockCurandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
