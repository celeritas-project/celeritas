//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GenericGeoTestBase.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<>
struct GenericGeoTraits<OrangeParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<OrangeStateData, M>;

    using TrackView = OrangeTrackView;
    static inline char const* ext = ".org.json";
    static inline char const* name = "ORANGE";
};

using OrangeTestBase = GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
