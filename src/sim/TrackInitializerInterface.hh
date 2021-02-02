//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "geometry/GeoInterface.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/Primary.hh"
#include "SimInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Lightweight version of a track used to initialize new tracks from primaries
 * or secondaries.
 */
struct TrackInitializer
{
    SimTrackState       sim;
    GeoStateInitializer geo;
    ParticleTrackState  particle;
};

//---------------------------------------------------------------------------//
/*!
 * View to the data used to initialize new tracks.
 */
struct TrackInitializerPointers
{
    Span<TrackInitializer>    initializers;
    Span<size_type>           parent;
    Span<size_type>           vacancies;
    Span<size_type>           secondary_counts;
    Span<TrackId::value_type> track_counter;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !secondary_counts.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
