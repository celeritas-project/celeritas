//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.cu
//---------------------------------------------------------------------------//
#include "Filler.device.t.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template struct Filler<real_type, MemSpace::device>;
template struct Filler<size_type, MemSpace::device>;
template struct Filler<int, MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
