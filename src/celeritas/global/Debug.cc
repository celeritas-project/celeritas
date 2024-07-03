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
    std::clog << j.dump(1);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * During an execute call, for interactive debugging, a pointer.
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
/*!
 * Print a SimTrackView on host to std::clog.
 */
void debug_print(SimTrackView const& view)
{
    debug_print_impl(view);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
