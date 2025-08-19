//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Debug.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>

namespace celeritas
{
class OrangeTrackView;

//---------------------------------------------------------------------------//
//! Print an ORANGE track view to a stream
struct StreamableOrangeTrack
{
    OrangeTrackView const& track;
};

// Write the track to a stream
std::ostream& operator<<(std::ostream&, StreamableOrangeTrack const&);

// Get a string JSON representation of an ORANGE track view
std::string to_json_string(OrangeTrackView const&);

//---------------------------------------------------------------------------//
// Print to clog everything that can be printed about an ORANGE track view
void debug_print(OrangeTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
