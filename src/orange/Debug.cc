//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Debug.cc
//---------------------------------------------------------------------------//
#include "Debug.hh"

#include <iostream>
#include <nlohmann/json.hpp>

#include "DebugIO.json.hh"
#include "OrangeTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream&
operator<<(std::ostream& os, StreamableOrangeTrack const& track_wrap)
{
    nlohmann::json j = track_wrap.track;
    os << j.dump();
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Get a string JSON representation of an ORANGE track view.
 */
std::string to_json_string(OrangeTrackView const& view)
{
    nlohmann::json j = view;
    return j.dump();
}

//---------------------------------------------------------------------------//
void debug_print(OrangeTrackView const& view)
{
    nlohmann::json j = view;
    std::clog << j.dump(1) << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
