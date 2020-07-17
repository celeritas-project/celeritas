//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#ifdef __NVCC__
#    include <curand_kernel.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#ifdef __NVCC__
using RngState = curandState_t;
#endif
using seed_type = unsigned long long;

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
