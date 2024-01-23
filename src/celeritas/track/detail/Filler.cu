//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/Filler.cu
//---------------------------------------------------------------------------//
#include "corecel/data/detail/Filler.device.t.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template struct Filler<TrackStatus, MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
