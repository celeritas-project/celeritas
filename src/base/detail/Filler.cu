//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Filler.cu
//---------------------------------------------------------------------------//
#include "Filler.t.cuh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template class Filler<real_type, MemSpace::device>;
template class Filler<size_type, MemSpace::device>;
template class Filler<int, MemSpace::device>;
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
