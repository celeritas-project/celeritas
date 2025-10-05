//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Debug.hh
//! \brief Utilities for interactive debugging and diagnostic output
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "celeritas/geo/GeoFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Forward declarations
class CoreParams;
class CoreTrackView;
class ParticleTrackView;
class SimTrackView;

//---------------------------------------------------------------------------//
//! Print a track to a stream
struct StreamableTrack
{
    CoreTrackView const& track;
};

std::ostream& operator<<(std::ostream&, StreamableTrack const&);

//---------------------------------------------------------------------------//
// Print to clog everything that can be printed about a core track view
void debug_print(CoreTrackView const&);

//---------------------------------------------------------------------------//
// Print to clog a SimTrackView on host
void debug_print(SimTrackView const&);

//---------------------------------------------------------------------------//
// Print to clog a ParticleTrackView on host
void debug_print(ParticleTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
