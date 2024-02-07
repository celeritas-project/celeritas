//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GenericGeoTestBase.hh"
#include "geocel/g4/GeantGeoData.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "geocel/g4/GeantGeoTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeantGeoTrackView;

namespace test
{
//---------------------------------------------------------------------------//
template<>
struct GenericGeoTraits<GeantGeoParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<GeantGeoStateData, M>;
    using TrackView = GeantGeoTrackView;
    static inline char const* ext = ".gdml";
    static inline char const* name = "Geant4";
};

using GeantGeoTestBase = GenericGeoTestBase<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
