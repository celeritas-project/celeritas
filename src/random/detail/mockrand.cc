//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file mockrand.cc
//---------------------------------------------------------------------------//
#include "mockrand.hh"

#include "base/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void mockrand_init(unsigned long long,
                   unsigned long long,
                   unsigned long long,
                   MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

unsigned int mockrand(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

float mockrand_uniform(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

double mockrand_uniform_double(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA");
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
