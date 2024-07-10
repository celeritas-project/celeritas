//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Debug.cc
//! \brief Define helper functions that are callable from a debugger
//---------------------------------------------------------------------------//
#include "Debug.hh"

#include <iostream>
#include <nlohmann/json.hpp>

#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

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
/*!
 * In the stepping loop, for interactive breakpoints, a debug pointer.
 *
 * This is accessible when:
 * - Inside an \c execute call (i.e., during stepping)
 * - Using CoreParams on host
 * - The status checker is enabled
 *
 * ... or if running inside a unit test that sets them.
 *
 * \warning This is not thread safe: it should only be used in single-threaded
 * (or track-parallel) execution modes, and ONLY inside an interactive
 * debugger. See celeritas/track/Debug.hh .
 */
CoreParams const* g_debug_executing_params{nullptr};

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
