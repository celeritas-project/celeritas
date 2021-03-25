//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/PlacedVolume.h>
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct VGTraits;

template<>
struct VGTraits<MemSpace::host>
{
    using PlacedVolume = vecgeom::cxx::VPlacedVolume;
};

template<>
struct VGTraits<MemSpace::device>
{
    using PlacedVolume = vecgeom::cuda::VPlacedVolume;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
