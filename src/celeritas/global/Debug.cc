//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Debug.cc
//! \brief Define helper functions that are callable from a debugger
//---------------------------------------------------------------------------//
#include "Debug.hh"

#include <iostream>
#include <nlohmann/json.hpp>

#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "CoreTrackView.hh"
#include "DebugIO.json.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
void debug_print_impl(T const& view)
{
    nlohmann::json j = view;
    std::clog << j.dump(1) << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, StreamableTrack const& track_wrap)
{
    nlohmann::json j = track_wrap.track;
    os << j.dump();
    return os;
}

//---------------------------------------------------------------------------//
#define DEFINE_DEBUG_PRINT(TYPE)       \
    void debug_print(TYPE const& view) \
    {                                  \
        debug_print_impl(view);        \
    }

//!@{
//! Print a host view to std::clog.
DEFINE_DEBUG_PRINT(CoreTrackView)
DEFINE_DEBUG_PRINT(SimTrackView)
DEFINE_DEBUG_PRINT(ParticleTrackView)
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
