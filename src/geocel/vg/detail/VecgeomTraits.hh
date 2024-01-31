//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/VecgeomTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct VecgeomTraits;

template<>
struct VecgeomTraits<MemSpace::host>
{
    using PlacedVolume = vecgeom::cxx::VPlacedVolume;
};

template<>
struct VecgeomTraits<MemSpace::device>
{
    using PlacedVolume = vecgeom::cuda::VPlacedVolume;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
