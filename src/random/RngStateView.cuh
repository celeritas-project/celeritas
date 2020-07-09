//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateView.cuh
//---------------------------------------------------------------------------//
#ifndef random_RngStateView_cuh
#define random_RngStateView_cuh

#include "base/Types.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device view to a vector of CUDA random number generator states.
 *
 * This "view" is expected to be an argument to a kernel launch.
 */
struct RngStateView
{
    ssize_type size = 0;
    RngState* rng   = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // random_RngStateView_cuh
