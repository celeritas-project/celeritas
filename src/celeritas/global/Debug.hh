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
//! Print a track to the given stream
struct StreamableTrack
{
    CoreTrackView const& track;
};

std::ostream& operator<<(std::ostream&, StreamableTrack const&);

//---------------------------------------------------------------------------//
// Print everything that can be printed about a core track view
void debug_print(CoreTrackView const&);

//---------------------------------------------------------------------------//
// Print a SimTrackView on host
void debug_print(SimTrackView const&);

//---------------------------------------------------------------------------//
// Print a ParticleTrackView on host
void debug_print(ParticleTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
