//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ThrustConfig.hh
//! \brief Platform and version-specific thrust setup
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/version.h>

namespace celeritas
{
#if CELERITAS_USE_CUDA
namespace thrust_native = thrust::cuda;
#elif CELERITAS_USE_HIP
namespace thrust_native = thrust::hip;
#endif

inline constexpr auto& thrust_execution_policy()
{
#if THRUST_MAJOR_VERSION == 1 && THRUST_MINOR_VERSION < 16
    return thrust_native::par;
#else
    return thrust_native::par_nosync;
#endif
}
//---------------------------------------------------------------------------//
}  // namespace celeritas