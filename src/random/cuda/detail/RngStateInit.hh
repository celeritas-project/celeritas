//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "random/cuda/RngStatePointers.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Initialize the RNG state on device
void rng_state_init_device(const RngStatePointers&         device_ptrs,
                           Span<const RngSeed::value_type> device_seeds);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
