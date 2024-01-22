//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/mockrand.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"

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
//! Random functions with cuRAND-like interface.
inline void mockrand_init(unsigned long long,
                          unsigned long long,
                          unsigned long long,
                          MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline unsigned int mockrand(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline float mockrand_uniform(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline double mockrand_uniform_double(MockRandState*)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
