//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VecgeomTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/GenericGeoTestBase.hh"
#include "celeritas/ext/VecgeomData.hh"
#include "celeritas/ext/VecgeomParams.hh"
#include "celeritas/ext/VecgeomTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<>
struct GenericGeoTraits<VecgeomParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<VecgeomStateData, M>;
    using TrackView = VecgeomTrackView;
    static inline char const* ext = ".gdml";
    static inline char const* name = "VecGeom";
};

using VecgeomTestBase = GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
