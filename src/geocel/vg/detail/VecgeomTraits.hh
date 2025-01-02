//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomTraits.hh
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
